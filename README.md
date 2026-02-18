# Cognitive-Model-Health-A-Prescriptive-Monitoring-Audit-System
It is a Cognitive BI system for Model Health Monitoring designed to detect "silent failures" in ML models. It uses prescriptive logic to analyze data drift and performance decay. By identifying reliability gaps early, it provides actionable insights and retraining strategies to ensure model integrity.
Key Cognitive Features
Prescriptive Reasoning Engine: The system goes beyond reporting "what" happened; it utilizes a diagnostic logic layer to prescribe specific corrective actions (e.g., sliding-window retraining) before model failure occurs.

Leading-Indicator Drift Detection: By tracking feature-level distribution shifts (Data Drift), the system predicts future performance drops before they impact the business bottom line.

Temporal Integrity Audit: Identifies "Environmental Aging" by monitoring the drift of time-based features (like TransactionDT), signaling when a model has become a "temporal stranger" to its current data stream.

Chaos-Mode Stress Testing: Includes a synthetic noise injection engine to validate the robustness of the monitoring triggers and simulate adversarial data environments.
echnical Architecture

Core Logic: Built with Python and Scikit-Learn, utilizing ROC-AUC as the primary health metric and Mean Shift Analysis for drift quantification.

Observability Interface: An ultra-modern Streamlit dashboard featuring custom CSS-injected glassmorphism and real-time Plotly visualizations.

Data Persistence: Automated audit logging via a CSV-based historical engine to track model performance and drift trends over time.

Dynamic Preprocessing: Features an adaptive pipeline that handles log-transformations, categorical encoding, and scaling based on a pre-configured Model Registry.

Achievement & Impact
Eliminated Silent Failures: Established a proactive alert system using a 20% drift tolerance threshold to catch model decay early.

Strategic Diagnostics: Developed a "Top Offender" identification logic that automatically highlights which specific feature (e.g., TransactionDT) is the primary driver of model instability.

Audit Portability: Integrated a report generation feature allowing stakeholders to download structured audit logs for regulatory compliance and model governance.
