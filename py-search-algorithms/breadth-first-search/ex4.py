# EXPERIMENT - 4

# Bayesian network model 
# pip install pgmpy

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

# Define the structure of the Bayesian Network
model = BayesianModel([('A', 'C'), ('B', 'C'), ('B', 'D'), ('C', 'E')])

# Define the CPDs
cpd_A = TabularCPD('A', 2, [[0.6], [0.4]])
cpd_B = TabularCPD('B', 2, [[0.7], [0.3]])

# C has two parents: A and B -> So evidence_card = [2, 2]
cpd_C = TabularCPD('C', 2,
                   [[0.9, 0.8, 0.7, 0.1],
                    [0.1, 0.2, 0.3, 0.9]],
                   evidence=['A', 'B'],
                   evidence_card=[2, 2])

cpd_D = TabularCPD('D', 2,
                   [[0.95, 0.2],
                    [0.05, 0.8]],
                   evidence=['B'],
                   evidence_card=[2])

cpd_E = TabularCPD('E', 2,
                   [[0.99, 0.1],
                    [0.01, 0.9]],
                   evidence=['C'],
                   evidence_card=[2])

# Add CPDs to the model
model.add_cpds(cpd_A, cpd_B, cpd_C, cpd_D, cpd_E)

# Validate the model
print("Model is valid:", model.check_model())
