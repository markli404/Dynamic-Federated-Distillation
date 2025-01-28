import numpy as np

parenting_stress = np.random.normal(loc=5, scale=3, size=100)


personal_well_being_error = np.random.normal(loc=0, scale=1, size=100)
social_support_error = np.random.normal(loc=0, scale=5, size=100)
child_behavioral_problems_error  = np.random.normal(loc=0, scale=25, size=100)

personal_well_being = parenting_stress / 5 + personal_well_being_error
social_support = parenting_stress * 2+ 4 + social_support_error
child_behavioral_problems= parenting_stress * 10 + 57 + child_behavioral_problems_error

data = np.vstack([parenting_stress, personal_well_being, social_support, child_behavioral_problems]).T

correlation_matrix = np.corrcoef(data, rowvar=False)

print(0)
