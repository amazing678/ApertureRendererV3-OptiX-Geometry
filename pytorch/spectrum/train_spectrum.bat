call C:/Users/admin/anaconda3/Scripts/activate.bat NeuralGame
python train_spectrum.py ^
--batch-size 8192 ^
--num-workers 16 ^
--log-every 100 ^
--eval-every 2000 ^
--ckpt-every 20000 ^
--restore-path "G:\Pytorch\ApertureRenderer\pytorch\spectrum\results\best.pth"
pause
