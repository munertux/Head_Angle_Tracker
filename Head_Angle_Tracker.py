import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import pyautogui
import Metodos


hipotenusa=Metodos.hipotenusa
width, height = pyautogui.size()
x_percent = 50
y_percent = 50

limite_de=0
limite_iz=0
mp_face_mesh = mp.solutions.face_mesh
pyautogui.PAUSE = 0    
cap = cv.VideoCapture(0)


if __name__ == "__main__":            
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.flip(frame, 1)
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                mesh_points = np.array([
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark
                ])
                coord_oi = None
                coord_od = None
                nariz = None
                frente = None
                menton = None
                for i, point in enumerate(mesh_points):                
                    if i == 323:
                        coord_oi = mesh_points[i]  #coordenada oreja izquierda
                    elif i == 93:
                        coord_od = mesh_points[i]#coordenada oreja derecha
                    elif i == 1:
                        nariz = mesh_points[i]#coordenada nariz
                    elif i == 10:
                        frente = mesh_points[i]#coordenada frente
                    elif i == 152:
                        menton = mesh_points[i]#coordenada menton                                          
                if coord_oi is not None and coord_od is not None:
                    # la variable origen contiene las coordenadas (x,y) del origen del plano o del origen de la cabeza del usuario
                    origen = [coord_oi[0] - int((coord_oi[0] - coord_od[0]) / 2), coord_oi[1] - int((coord_oi[1] - coord_od[1]) / 2)]
                    # la variable rotZ contiene la rotacion del plano de acuerdo al eje z de la cabeza del sujeto
                    rotZ = math.atan(abs((coord_oi[1] - coord_od[1]) / (coord_oi[0] - coord_od[0])))
                    # Aquí se suma 90 grados al ángulo rotZ y se guarda en la variable dis_Z
                    #la variable dis_z es la encargada de darle el angulo para dibujar con openCV el eje Y de la cabeza
                    if coord_oi[1] > coord_od[1]:
                        rotZ = -rotZ                
                    dis_Z = (math.pi / 2) + rotZ 
                    rotZ = math.degrees(rotZ)

                    # la variable dis_frente y dis_menton contienen la distancia del origen a la frente y al menton
                    dis_frente = (hipotenusa(menton,frente))/2
                    dis_menton = (hipotenusa(menton,frente))/2
                    # ajustamos las coordenadas del ejeY para que se ajuste al origen
                    sup_Y = [origen[0] + int(math.cos(dis_Z) * dis_frente), origen[1] - int(math.sin(dis_Z) * dis_frente)]
                    inf_Y = [origen[0] - int(math.cos(dis_Z) * dis_menton), origen[1] + int(math.sin(dis_Z) * dis_menton)]
                    #se dibuja con opencv ejes x,y,z en la cabeza del usuario
                    cv.line(frame, origen, sup_Y, (0, 255, 0), 3)
                    cv.line(frame, origen, inf_Y, (0, 255, 0), 3)
                    cv.line(frame, coord_od, coord_oi, (0, 0, 255), 3)
                    cv.line(frame, origen, nariz, (255, 0, 0), 3)
                    #calculos para determinar angulo de rotacion del plano en el eje "x" y "y" segun rotacion de la cabeza del usuario
                    cen_na = [abs(origen[0]-nariz[0]),abs(origen[1]-nariz[1])]                    
                    cen_or = (hipotenusa(coord_oi,coord_od))/ 2
                    rotY = -(cen_na[0] * 45 / cen_or)
                    if origen[1] < nariz[1]:
                        rotX = -(cen_na[1] * 45 / dis_menton)
                    else:
                        rotX = (cen_na[1] * 45 / dis_frente)
                    if nariz[0] < origen[0]:
                        rotY = -(rotY)
                    #Mostrar en pantalla Angulos de rotacion
                    cv.putText(frame, "Rotacion eje X=" + str(rotX), [10, 20], cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
                    cv.putText(frame, "Rotacion eje Y=" + str(rotY),  [10, 40], cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
                    cv.putText(frame, "Rotacion eje Z=" + str(rotZ), [10, 60], cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)                
            cv.imshow('img', frame)
            key = cv.waitKey(1)
            if key == ord('q'):
                break  
    cap.release()
    cv.destroyAllWindows()
