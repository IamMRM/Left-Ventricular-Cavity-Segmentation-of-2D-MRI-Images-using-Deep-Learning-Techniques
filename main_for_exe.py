import predict_compact_unet_for_exe
import predict_compact_fcn_for_exe
import cv2
import numpy as np

while True:
	a = input("Enter path of NIFTI series; to quit press 'q'")
	if a == 'q':
		break
	while True:
		b = input("Press (1) for U-Net results, (2) for FCN results, (3) for both")
		if b == '1':
			pred = predict_compact_unet_for_exe.predict(a)
			print(pred.shape)
			cv2.imshow("U-Net - press escape to exit", pred)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			cv2.imwrite("result_unet.jpeg")
			break
		elif b == '2':
			pred = predict_compact_fcn_for_exe.predict(a)
			print(pred.shape)
			cv2.imshow("FCN - press escape to exit", pred)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			cv2.iwrite("result_fcn.jpeg", pred)
			break
		elif b == '3':
			pred_0 = predict_compact_fcn_for_exe.predict(a)
			pred_1 = predict_compact_unet_for_exe.predict(a)
			# pred = np.vstack((pred_0, pred_1))
			print(pred_0.shape)
			cv2.imshow("FCN - press escape for next", pred_0)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			cv2.imwrite("result_fcn.jpeg", pred_0)
			print(pred_1.shape)
			cv2.imshow("U-Net - press escape to exit", pred_1)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			cv2.imwrite("result_fcn.jpeg", pred_1)
			break





