import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "acl_mpl_cache"))


class RiskExplainer:
    """
    Wraps SHAP TreeExplainer for the XGBoost ACL risk model.
    Converts raw SHAP values into:
      - shap_values dict  (for the SHAP bar chart in the UI)
      - human-readable explanations
      - highlight_joints dict  (joint → BGR color for the video overlay)
    """

    def __init__(self, model):
        import shap

        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def explain_prediction(self, features_dict: dict, display_features: dict | None = None) -> dict:
        import pandas as pd

        df = pd.DataFrame([features_dict])

        try:
            raw = self.explainer.shap_values(df)
            # shap_values shape: (n_samples, n_features) for XGBoost binary
            if isinstance(raw, list):
                # Old SHAP: list of arrays, one per class → take class-1
                instance_shap = raw[1][0]
            else:
                instance_shap = raw[0]

            feature_names = df.columns.tolist()
            shap_dict = {f: float(v) for f, v in zip(feature_names, instance_shap)}
            sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

        except Exception as exc:
            # Graceful fallback: no SHAP values, skip chart but keep explanations
            shap_dict = {}
            sorted_shap = []

        result = self.fallback_explanation(display_features or features_dict)
        explanations = result['explanations']
        highlight_joints = result['highlight_joints']

        is_static = (display_features or features_dict).get('joint_velocity', 0.0) < 0.06

        # ── SHAP-based top contributor (if available) appended as insight ──
        if sorted_shap and not is_static:
            top_feat, top_val = sorted_shap[0]
            direction = "raises risk" if top_val > 0 else "lowers risk"
            label = top_feat.replace('_', ' ').title()
            explanations.insert(0, f"XAI top driver: {label} {direction} ({top_val:+.3f}).")
        elif is_static:
            shap_dict = {}
            explanations.insert(0, "Static stance detected; landing-risk model is held low until movement starts.")
        elif 'exc' in locals():
            explanations.insert(0, "XAI fallback: biomechanical rules shown while SHAP is unavailable.")

        return {
            'shap_values': shap_dict,
            'explanations': explanations[:4],   # cap at 4 lines
            'highlight_joints': highlight_joints,
        }

    @staticmethod
    def fallback_explanation(features_dict: dict) -> dict:
        explanations = []
        highlight_joints = {}

        RED    = (0,   0,   255)   # BGR red   = danger
        ORANGE = (0,   165, 255)   # BGR orange = warning
        GREEN  = (0,   255, 0  )   # BGR green  = good

        feat_vals = features_dict  # shorthand
        velocity = feat_vals.get('joint_velocity', 0.0)

        # ── Rule-based explanations (always shown, based on actual values) ──
        # These fire regardless of SHAP so there's always meaningful text.

        if feat_vals.get('left_knee_valgus', 0) > 10:
            explanations.append(
                f"Left knee valgus {feat_vals['left_knee_valgus']:.1f}° increases ACL risk."
            )
            highlight_joints['LEFT_KNEE'] = RED

        if feat_vals.get('right_knee_valgus', 0) > 10:
            explanations.append(
                f"Right knee valgus {feat_vals['right_knee_valgus']:.1f}° increases ACL risk."
            )
            highlight_joints['RIGHT_KNEE'] = RED

        if velocity > 0.12 and feat_vals.get('left_knee_flexion', 180) < 25:
            explanations.append(
                f"Stiff left landing ({feat_vals['left_knee_flexion']:.1f}° flexion) increases risk."
            )
            highlight_joints.setdefault('LEFT_KNEE', ORANGE)

        if velocity > 0.12 and feat_vals.get('right_knee_flexion', 180) < 25:
            explanations.append(
                f"Stiff right landing ({feat_vals['right_knee_flexion']:.1f}° flexion) increases risk."
            )
            highlight_joints.setdefault('RIGHT_KNEE', ORANGE)

        if feat_vals.get('knee_asymmetry', 0) > 15:
            explanations.append(
                f"Bilateral knee asymmetry {feat_vals['knee_asymmetry']:.1f}° increases risk."
            )
            highlight_joints.setdefault('LEFT_KNEE',  ORANGE)
            highlight_joints.setdefault('RIGHT_KNEE', ORANGE)

        if velocity > 0.15:
            explanations.append(
                f"High landing velocity ({velocity:.3f}) increases risk."
            )

        if not explanations:
            if velocity < 0.06:
                explanations.append("Standing or slow movement detected; monitor alignment for the next landing.")
            else:
                explanations.append("Good lower-body alignment; low ACL risk detected.")
            highlight_joints['LEFT_KNEE']  = GREEN
            highlight_joints['RIGHT_KNEE'] = GREEN

        return {
            'shap_values':     {},
            'explanations':    explanations,
            'highlight_joints': highlight_joints,
        }
