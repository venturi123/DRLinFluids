from drlinfluids.runner import run
import os

def test_run():
    foam_params = {}
    foam_params['verbose'] = False
    run(
        path=os.path.split(os.path.realpath(__file__))[0], 
        foam_params=foam_params, 
        deltaT=0.1, 
        start_time=0, 
        end_time=0.1
    )

