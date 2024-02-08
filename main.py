import EvoSystem

EvoSystem.StartSim(
    AmountOfGen = 50,
    Seed = 20,
    UseLinReg = 1, # True for LinReg false for Sine
    UseCrowding = 2, # 0 for no, 1 for DeJong, And 2 for deterministic
    UseReplacementSelection = 0,
    UseFitnessSelection = 1,
    BitstringLength = 101, # always 101 for LinReg
    MutationRate = 1,
    CrossoverRate = 0.7,
    PopSize = 30,
    AmountOfParents = 30, #Always even
    Constraint = None
)