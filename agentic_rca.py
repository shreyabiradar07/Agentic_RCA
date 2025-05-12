import random
import os

from transformers import pipeline

LLM_MODEL_NAME = "google/flan-t5-large"  # Example model


class ObservationAgent:
    def detect_frontend_slowdown(self):
        """Simulates monitoring frontend performance and detecting a slowdown."""
        # Force a slowdown for testing purposes
        return {"frontend_service": "WebFrontend", "timestamp": "2025-05-09T10:00:00Z"}

    def _is_slowdown_detected(self):
        # Simulate a condition for slowdown (e.g., based on a random number)
        import random
        return random.random() < 0.2  # 20% chance of slowdown in this simulation


class ContextualizationAgent:
    def __init__(self, knowledge_graph, logging_data, monitoring_data_simulated):
        self.knowledge_graph = knowledge_graph
        self.logging_data = logging_data
        self.monitoring_data_simulated = monitoring_data_simulated

    def gather_initial_context(self, trigger_info):
        frontend_service = trigger_info["frontend_service"]
        dependencies = self._get_dependencies(frontend_service)
        return {"frontend_service": frontend_service, "dependencies": dependencies}

    def _get_dependencies(self, service):
        return self.knowledge_graph.get(service, {}).get("depends_on", [])

    def fetch_relevant_metrics_and_logs(self, context):
        frontend_service = context["frontend_service"]
        # dependencies = context["dependencies"]
        all_dependencies = self._collect_all_dependencies(frontend_service)
        metrics = self._get_dynamic_metrics(frontend_service, all_dependencies)


        # metrics = self._get_dynamic_metrics(frontend_service, dependencies)  # Updated
        logs = self._simulate_logs(frontend_service, all_dependencies)

        # for svc, vals in metrics.items():
        #     print(f"[METRIC] {svc}: " + ", ".join(f"{k} = {v}" for k, v in vals.items()))

        return {"metrics": metrics, "logs": logs}

    def _get_dynamic_metrics(self, service, dependencies):
        def extract_latency(entry):
            latency = round(
                entry.get("p99_latency") or
                entry.get("response_time") or
                entry.get("query_execution_time") or
                random.uniform(0.2, 0.8), 2
            )
            return latency

        services = [service] + dependencies
        metrics = {}
        for s in services:
            entry = self.monitoring_data_simulated.get(s, {})
            latency = extract_latency(entry)
            print(f"[METRIC] {s}: latency = {latency}")  # ← Add this line
            metrics[s] = {"latency": latency}
        return metrics

    def _collect_all_dependencies(self, service, visited=None):
        if visited is None:
            visited = set()
        for dep in self.knowledge_graph.get(service, {}).get("depends_on", []):
            if dep not in visited:
                visited.add(dep)
                self._collect_all_dependencies(dep, visited)
        return list(visited)





    def _simulate_logs(self, service, dependencies):
        logs = {service: []}
        for dep in dependencies:
            if "DatabaseX" in dep:
                logs[dep] = ["Slow query detected"]  # Simulate slow query in DatabaseX
            else:
                logs[dep] = []
        return logs


class HypothesisGenerationAgent:
    def __init__(self, knowledge_graph, use_llm=False):
        self.knowledge_graph = knowledge_graph
        self.llm = None
        if use_llm:
            try:
                self.llm = pipeline("text2text-generation", model=LLM_MODEL_NAME)
            except Exception as e:
                print(f"Error initializing LLM in HypothesisGenerationAgent: {e}")

    def generate_hypotheses(self, context, metrics_logs):
        frontend_service = context["frontend_service"]
        dependencies = context["dependencies"]
        initial_hypotheses = self._graph_traversal_hypotheses(frontend_service)
        llm_suggestions = self._llm_powered_suggestions(frontend_service, dependencies,
                                                        metrics_logs) if self.llm else []
        return self._prioritize_hypotheses(initial_hypotheses + llm_suggestions)

    def _graph_traversal_hypotheses(self, start_service):
        """Generates hypotheses by traversing the knowledge graph."""
        hypotheses = []
        queue = [(start_service, [start_service])]
        visited = {start_service}
        while queue:
            current_service, path = queue.pop(0)
            if "database" in current_service.lower():
                hypotheses.append(f"Potential slow query in {current_service}")
            for relation in self.knowledge_graph.get(current_service, {}):
                if relation in ["depends_on", "calls"]:
                    for neighbor in self.knowledge_graph[current_service][relation]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            new_path = list(path)
                            new_path.append(neighbor)
                            queue.append((neighbor, new_path))
        return hypotheses

    def _llm_powered_suggestions(self, frontend_service, dependencies, metrics_logs):
        if self.llm:
            prompt = f"The frontend service '{frontend_service}' is slow because its backend dependency '{dependencies[0]}' might be slow. Suggest up to 2 concise and distinct potential underlying causes for performance issues in these backend services or their communication."
            try:
                response = self.llm(prompt, max_length=80, num_return_sequences=1, do_sample=True, top_k=30, top_p=0.8)
                suggestion = response[0]['generated_text'].strip()
                if suggestion and "faulty service key" not in suggestion.lower():
                    return [f"(LLM Suggestion) {suggestion}"]
            except Exception as e:
                print(f"Error calling Hugging Face model for suggestions: {e}")
            return []
        return []

    def _prioritize_hypotheses(self, hypotheses):
        """Prioritizes the generated hypotheses."""
        # Simple prioritization based on keywords for now. More sophisticated logic can be added.
        prioritized = []
        database_related = [h for h in hypotheses if "database" in h.lower()]
        other_related = [h for h in hypotheses if h not in database_related]
        prioritized.extend(database_related)
        prioritized.extend(other_related)
        return prioritized


class EvidenceGatheringAgent:
    def __init__(self, knowledge_graph, logging_data, monitoring_data_simulated):
        self.knowledge_graph = knowledge_graph
        self.logging_data = logging_data
        self.monitoring_data_simulated = monitoring_data_simulated

    def gather_evidence(self, prioritized_hypotheses, context, metrics_logs):
        """Gathers detailed evidence for the suspected services and databases."""
        evidence = {}
        suspects = set()
        for hypothesis in prioritized_hypotheses:
            if "database" in hypothesis.lower():
                db_name = hypothesis.split("in ")[-1]
                suspects.add(db_name)
            # Add logic to identify other suspect services based on hypotheses

        # Also gather evidence for the dependencies of the frontend
        for dependency in context.get("dependencies", []):
            suspects.add(dependency)

        for suspect in suspects:
            metrics = self._get_metrics(suspect, metrics_logs)  # Use metrics_logs
            logs = self._analyze_logs(suspect)
            evidence[suspect] = {"metrics": metrics, "logs": logs}

        return evidence

    def _get_metrics(self, service, metrics_logs):
        """Retrieves metrics from the provided metrics_logs."""
        return metrics_logs.get(service, self.monitoring_data_simulated.get(service, {}))

    def _get_detailed_metrics(self, service):
        """Queries monitoring systems for granular metrics (currently unused but kept)."""
        return self.monitoring_data_simulated.get(service, {}).get("detailed", {})

    def _analyze_logs(self, service):
        """Analyzes logs for error messages, slow queries, etc."""
        return self.logging_data.get(service, [])


class ReasoningCorrelationAgent:
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph

    def correlate_evidence(self, context, evidence):
        reasoning = []
        if evidence.get("DatabaseX", {}).get("logs", []) and "Slow query detected" in evidence["DatabaseX"]["logs"] and \
                evidence.get("ServiceA", {}).get("metrics", {}).get("response_time", 0) > 0.5:
            return {"root_cause": "Slow query in DatabaseX causing high latency in ServiceA.",
                    "reasoning": ["High latency in ServiceA", "Slow query detected in DatabaseX"]}
        elif evidence.get("ServiceA", {}).get("metrics", {}).get("response_time", 0) > 0.5:
            return {"root_cause": "High latency in ServiceA is impacting the frontend.",
                    "reasoning": ["High latency in ServiceA"]}
        return {"root_cause": "Could not definitively determine the root cause with the current evidence.",
                "reasoning": []}


class ExplanationReportingAgent:
    def __init__(self, use_llm=False):
        self.llm = None
        if use_llm:
            try:
                self.llm = pipeline("text2text-generation", model=LLM_MODEL_NAME)
            except Exception as e:
                print(f"Error initializing LLM in ExplanationReportingAgent: {e}")

    def generate_report(self, root_cause_analysis, context, evidence):
        root_cause = root_cause_analysis.get("root_cause", "Unknown")
        reasoning = root_cause_analysis.get("reasoning", [])
        explanation = f"The likely root cause of the frontend slowdown on '{context['frontend_service']}' is: {root_cause}. "
        if reasoning and self.llm:
            prompt = ""
            if "High latency in ServiceA" in root_cause:
                prompt = f"The frontend is slow, and its dependency 'ServiceA' has high latency. Suggest several specific troubleshooting steps solving the issue faced."
                if "Slow query in DatabaseX" in root_cause:
                    prompt = f"The frontend is slow, 'ServiceA' has high latency, likely due to slow queries in its database 'DatabaseX'. List specific troubleshooting actions for slow database queries and backend code that interacts with the database."
            else:
                prompt = f"The frontend is slow. Based on the evidence: '{', '.join(reasoning)}', suggest specific troubleshooting actions for a frontend slowdown in a multi-service architecture."

            if prompt:
                try:
                    response = self.llm(prompt, max_length=200, num_return_sequences=1)
                    explanation += "Possible Solutions: " + response[0]['generated_text'].strip()
                except Exception as e:
                    explanation += f"Reasoning: {', '.join(reasoning)}"
            else:
                explanation += f"Reasoning: {', '.join(reasoning)}"
        else:
            explanation += f"Reasoning: {', '.join(reasoning)}"
        return {"report": explanation}


class FeedbackAgent:
    def gather_feedback(self, report):
        """Simulates gathering human feedback on the report."""
        # In a real system, this would involve a human reviewing the report.
        # For simulation, we'll just print the report and assume it's reviewed.
        print("Generated Report:")
        print(report["report"])
        feedback = input("Is this analysis correct? (yes/no/provide correction): ")
        return feedback

    def update_knowledge(self, feedback, current_knowledge):
        """Simulates updating the knowledge graph based on feedback."""
        if "no" in feedback.lower() or "correction" in feedback.lower():
            print("Feedback received. Knowledge graph and rules would be updated in a real system.")
            # In a real system, you would implement logic to update the knowledge graph
            # and potentially retrain any LLM components based on the feedback.
        else:
            print("Feedback indicates the analysis was correct.")
        return current_knowledge  # Return the (potentially updated) knowledge


# class RCAAgent:
#     def __init__(self, knowledge_graph, monitoring_config, logging_data, llm=None):
#         self.observation_agent = ObservationAgent()
#         self.contextualization_agent = ContextualizationAgent(knowledge_graph, monitoring_config, logging_data)
#         self.hypothesis_generation_agent = HypothesisGenerationAgent(knowledge_graph, llm)
#         self.evidence_gathering_agent = EvidenceGatheringAgent(monitoring_config, logging_data)
#         self.reasoning_correlation_agent = ReasoningCorrelationAgent(knowledge_graph) # Pass knowledge_graph here
#         self.explanation_reporting_agent = ExplanationReportingAgent()
#         self.feedback_agent = FeedbackAgent()
#         self.knowledge_graph = knowledge_graph
#
#
#     def run_rca(self):
#         """Orchestrates the root cause analysis workflow."""
#         trigger_info = self.observation_agent.detect_frontend_slowdown()
#         if trigger_info:
#             print("Frontend slowdown detected. Initiating RCA...")
#             # Gather initial context using the ContextualizationAgent
#             context = self.contextualization_agent.gather_initial_context(trigger_info)
#             metrics_logs = self.contextualization_agent.fetch_relevant_metrics_and_logs(context)
#             hypotheses = self.hypothesis_generation_agent.generate_hypotheses(context, metrics_logs)
#             print(f"Generated Hypotheses: {hypotheses}")
#             evidence = self.evidence_gathering_agent.gather_evidence(hypotheses, context)
#             root_cause_analysis = self.reasoning_correlation_agent.correlate_evidence(context, evidence)
#             report = self.explanation_reporting_agent.generate_report(root_cause_analysis)
#             feedback = self.feedback_agent.gather_feedback(report)
#             self.knowledge_graph = self.feedback_agent.update_knowledge(feedback, self.knowledge_graph)
#         else:
#             print("No frontend slowdown detected.")


class RCAAgent:
    def __init__(self, knowledge_graph, logging_data, monitoring_data_simulated, use_llm=None):
        self.observation_agent = ObservationAgent()
        self.contextualization_agent = ContextualizationAgent(knowledge_graph, logging_data, monitoring_data_simulated)
        self.hypothesis_generation_agent = HypothesisGenerationAgent(knowledge_graph, use_llm)
        self.evidence_gathering_agent = EvidenceGatheringAgent(knowledge_graph, logging_data, monitoring_data_simulated)
        self.reasoning_correlation_agent = ReasoningCorrelationAgent(knowledge_graph)
        self.explanation_reporting_agent = ExplanationReportingAgent(use_llm)
        self.feedback_agent = FeedbackAgent()
        self.knowledge_graph = knowledge_graph

    def run_rca(self):
        """Orchestrates the root cause analysis workflow."""
        trigger_info = self.observation_agent.detect_frontend_slowdown()
        if trigger_info:
            print("Frontend slowdown detected. Initiating RCA...")
            # Gather initial context using the ContextualizationAgent
            context = self.contextualization_agent.gather_initial_context(trigger_info)
            metrics_logs = self.contextualization_agent.fetch_relevant_metrics_and_logs(context)
            hypotheses = self.hypothesis_generation_agent.generate_hypotheses(context, metrics_logs)
            print(f"Generated Hypotheses: {hypotheses}")
            evidence = self.evidence_gathering_agent.gather_evidence(hypotheses, context, metrics_logs)
            root_cause_analysis = self.reasoning_correlation_agent.correlate_evidence(context, evidence)
            report = self.explanation_reporting_agent.generate_report(root_cause_analysis, context, evidence)
            feedback = self.feedback_agent.gather_feedback(report)
            self.knowledge_graph = self.feedback_agent.update_knowledge(feedback, self.knowledge_graph)
        else:
            print("No frontend slowdown detected.")


# Simulate the knowledge graph, monitoring data, and logging data
# knowledge_graph_data = {
#     "WebFrontend": {"depends_on": ["ServiceA"], "alerts": []},
#     "ServiceA": {"depends_on": ["DatabaseX"], "calls": ["DatabaseX"], "alerts": [{"timestamp": "2025-05-09T09:45:00Z", "message": "High CPU usage"}], "recent_changes": [{"timestamp": "2025-05-08T18:00:00Z", "service": "ServiceA", "change": "Increased connection pool size"}]},
#     "DatabaseX": {"used_by": ["ServiceA"], "alerts": [{"timestamp": "2025-05-09T09:50:00Z", "message": "High query latency"}]}
# }

# monitoring_data_simulated = {
#     "WebFrontend": {"p99_latency": 0.8},
#     "ServiceA": {"response_time": 0.6},
#     "DatabaseX": {"query_execution_time": 1.2, "cpu_utilization": 0.7, "detailed": {"slowest_queries": ["SELECT * FROM users WHERE id = ? (1.1s)"]}}
# }

def generate_monitoring_data():
    return {
        "WebFrontend": {"p99_latency": round(random.uniform(0.6, 0.9), 2)},
        "ServiceA": {"response_time": round(random.uniform(0.4, 0.8), 2)},
        "DatabaseX": {
            "query_execution_time": round(random.uniform(0.8, 1.3), 2),
            "cpu_utilization": round(random.uniform(0.5, 0.8), 2),
            "detailed": {
                "slowest_queries": [random.choice([
                    "SELECT * FROM users WHERE id = ? (1.1s)",
                    "SELECT COUNT(*) FROM orders WHERE status = 'pending' (0.9s)",
                    "SELECT * FROM products WHERE price > 1000 (1.3s)"
                ])]
            }
        }
    }

logging_data_simulated = {
    "WebFrontend": ["Request to /api/data took 0.7s"],
    "ServiceA": ["Calling DatabaseX...", "DatabaseX response took 1.1s"],
    "DatabaseX": ["Slow query: SELECT * FROM users WHERE id = ? took 1.1s", "Connection established",
                  "Query completed in 0.01s"]
}

# Simulate the knowledge graph and logging data
knowledge_graph_data = {
    "WebFrontend": {"depends_on": ["ServiceA"]},
    "ServiceA": {"depends_on": ["DatabaseX"]},
    "DatabaseX": {"used_by": ["ServiceA"]}
}

monitoring_data_simulated = generate_monitoring_data()

# Initialize and run the RCA Agent
rca_agent = RCAAgent(knowledge_graph_data, logging_data_simulated, monitoring_data_simulated, use_llm=True)
rca_agent.run_rca()
