import petsctools

PETSc = petsctools.init()

appctx = petsctools.AppContext()
reynolds = 10

opts = petsctools.OptionsManager(
    parameters={
        'fieldsplit_1_pc_type': 'python',
        'fieldsplit_1_pc_python_type': 'firedrake.MassInvPC',
        'fieldsplit_1_mass_reynolds': appctx.add(reynolds)},
    options_prefix="")

with opts.inserted_options():
    re = appctx.get('fieldsplit_1_mass_reynolds')
    print(f"{re = }")
    re = appctx['fieldsplit_1_mass_reynolds']
    print(f"{re = }")

    re = appctx.get('fieldsplit_0_mass_reynolds', 20)
    print(f"{re = }")
    re = appctx['fieldsplit_0_mass_reynolds']
    print(f"{re = }")
