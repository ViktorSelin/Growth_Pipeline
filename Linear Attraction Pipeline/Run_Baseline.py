from GrowthPipeline import Initialize_Growth, Score_Growth_3D


baseline = True
seed = 0

growth = Initialize_Growth(seed,baseline=True)
eps = Score_Growth_3D(growth.nano,baseline=True,e_func=True)

print('Finished baseline')
