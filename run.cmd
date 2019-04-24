@echo off
%~d0
cd /d %~dp0

echo Semantic Segmentation: START...

:: generate dataset
::python ./tools/generate_dataset.py

:: color to index
::python ./tools/color2index.py

:: training segmentation network
::python ./examples/training.py

:: predicting
python ./examples/predicting.py

::evaluating
python ./examples/evaluating.py

pause
exit