cd %0/..
if not %ERRORLEVEL% == 0 goto end

labelme2yolo --json_dir . --val_size 0.5 --output_format bbox --label_list 10C 10D 10H 10S 2C 2D 2H 2S 3C 3D 3H 3S 4C 4D 4H 4S 5C 5D 5H 5S 6C 6D 6H 6S 7C 7D 7H 7S 8C 8D 8H 8S 9C 9D 9H 9S AC AD AH AS JC JD JH JS KC KD KH KS QC QD QH QS
if not %ERRORLEVEL% == 0 goto end

python determine.py
if not %ERRORLEVEL% == 0 goto end

python augment.py
if not %ERRORLEVEL% == 0 goto end

merge.cmd
if not %ERRORLEVEL% == 0 goto end

echo Everything is done!!

:end
pause
