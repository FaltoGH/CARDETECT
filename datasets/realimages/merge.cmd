cd %0/..
if not %ERRORLEVEL% == 0 goto end
copy .\YOLODataset\images\train\* "..\Playing Cards.v3-original_raw-images.yolov8\train\images\"
if not %ERRORLEVEL% == 0 goto end
copy .\YOLODataset\labels\train\* "..\Playing Cards.v3-original_raw-images.yolov8\train\labels\"
if not %ERRORLEVEL% == 0 goto end
copy .\YOLODataset\images\val\* "..\Playing Cards.v3-original_raw-images.yolov8\valid\images\"
if not %ERRORLEVEL% == 0 goto end
copy .\YOLODataset\labels\val\* "..\Playing Cards.v3-original_raw-images.yolov8\valid\labels\"
if not %ERRORLEVEL% == 0 goto end

:end
pause
