# import socket
# import cv2
# import numpy as np
# import threading



# def check_keyboard_input():
#     if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
#         command = sys.stdin.readline().strip()
#         return command
#     return ""

# # Function to receive data from sensor
# # def receive_data(sock):
# #     stop_flag = False
# #     while not stop_flag:
# #         command = check_keyboard_input()
# #         if command == "q":
# #             stop_flag = True
# #             continue
# #         data, addr = sock.recvfrom(1024)
# #         if not data:
# #             break   
# #         print("received message:", data.decode())
# #     sock.close()
# #     print("close")


# # UDP_IP = "0.0.0.0"

# # # Port for sensor data
# # data_port = 6000
# # data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# # data_sock.bind((UDP_IP, data_port))

# # # Start thread for receiving data
# # data_thread = threading.Thread(target=receive_data, args=(data_sock,))
# # data_thread.start()

# # # Open the camera using OpenCV
# # cap = cv2.VideoCapture(0)

# # # Display image in a loop
# # while True:
# #     ret, frame = cap.read()
# #     cv2.imshow("Image", frame)
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # # Release the camera and close the window
# # cap.release()
# # cv2.destroyAllWindows()


# import socket
# import cv2

# def receive_data(sock):
#     while True:
#         data, address = sock.recvfrom(1024)
#         print("Received data: ", data.decode())
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# sock.bind(("0.0.0.0", 6000))

# receive_data(sock)
# sock.close()



# # import socket
# # import cv2
# # import numpy as np

# # def receive_data(sock, data_shape, stop_flag):
# #     while True:
# #         # data, _ = sock.recvfrom(4096)
# #         data, addr = sock.recvfrom(1024)
# #         if data:
# #             # data = np.frombuffer(data, dtype=np.float32)
# #             # data = data.reshape(*data_shape)
# #             # 在這裡您可以進行其他的處理，例如將數據儲存到文件中
# #             print(data.decode())
# #             if stop_flag.is_set():
# #                 break

# # def run_camera(cap, stop_flag):
# #     while True:
# #         ret, frame = cap.read()
# #         if ret:
# #             # 在這裡您可以進行其他的處理，例如將影像顯示到畫面上
# #             cv2.imshow("camera", frame)
# #         if cv2.waitKey(1) & 0xFF == ord('q') or stop_flag.is_set():
# #             break

# # def main():
# #     HOST = "0.0.0.0"
# #     PORT = 6000
# #     data_shape = (100, 100)

# #     # 創建UDP socket
# #     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# #     sock.bind((HOST, PORT))

# #     # 開啟相機
# #     cap = cv2.VideoCapture(0)

# #     # 啟動多執行緒
# #     stop_flag = threading.Event()
# #     threading.Thread(target=receive_data, args=(sock, data_shape, stop_flag), daemon=True).start()
# #     threading.Thread(target=run_camera, args=(cap, stop_flag), daemon=True).start()

# #     # 等待結束
# #     input("Press any key to stop...")
# #     stop_flag.set()

# #     # 關閉相機
# #     cap.release()

# # if __name__ == "__main__":
# #     main()


## 背景執行的 Daemon Thread 在主程式被結束執行時，也會一同地被暴力結束執行。 


# import socket
# import cv2
# import numpy as np
# import threading
# import datetime
# import json
# from time import sleep

# def receive_data(sock, stop_flag):
#     try:
#         global mmwave_list
#         mmwave_list = []
#         while True:
#             data, _ = sock.recvfrom(2048)
#             if data:
#                 global mmwave_json

#                 mmwave_json = json.loads(data)
#                 mmwave_list.append(mmwave_json["TimeStamp"])
#                 # print(mmwave_json["TimeStamp"])


#                 # 在這裡您可以進行其他的處理，例如將數據儲存到文件中
#                 # print(data.decode())

#     except Exception as e:
#         # 捕獲異常
#         print('Thread worker failed:', e)
    


# def run_camera(cap, stop_flag):
#     while True:
#         ret, frame = cap.read()
#         if ret:
#             # 在這裡您可以進行其他的處理，例如將影像顯示到畫面上
#             cv2.imshow("camera", frame)
#             print(datetime.datetime.now().strftime("%H:%M:%S:%f"))
#             print(mmwave_list)
            
#         if cv2.waitKey(1) & 0xFF == ord('q') or stop_flag.is_set():
#             break

# def main():
#     HOST = "0.0.0.0"
#     PORT = 6000
#     data_shape = (100, 100)

#     # 創建UDP socket
#     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     sock.bind((HOST, PORT))

#     # 開啟相機
#     cap = cv2.VideoCapture(0)

#     # 啟動多執行緒
#     stop_flag = threading.Event()
#     threading.Thread(target=receive_data, args=(sock, stop_flag), daemon=True).start()
#     threading.Thread(target=run_camera, args=(cap, stop_flag), daemon=True).start()

#     sleep(3)
#     # 等待結束
#     input("Press any key to stop...")
#     stop_flag.set()

#     # 關閉相機
#     cap.release()

# if __name__ == "__main__":
#     main()


import socket
import threading
import cv2
import json
from datetime import datetime
import queue


def cal_time_error(t1, t2):
    # now_time = datetime.now()
    # current_time = now_time.time() # datetime.strptime(str(now_time), "%H%M%S%f")

    dT_now = datetime.combine(datetime.today(), t1) # combine the date 
    dT_mm = datetime.combine(datetime.today(), t2) # combine the date 
    
    time_error = (dT_now - dT_mm).total_seconds() # np.abs((dT_now - dT_mm).total_seconds()) # current_time-mmwave_time  # 0.015 second -> 15ms
    # print("Time Error:", time_error, "sec ", t1, t2)
    
    return time_error

def receive_data(udp_socket, BUFSIZE):
    while stop_program:
        data, addr = udp_socket.recvfrom(BUFSIZE)
        # 對收到的資料進行處理
        process_data(data)

def process_data(data):
    # 進行資料處理的實際代碼
    global mmwave_json
    global pre_mmwave_json
    if data:
        if mmwave_json != "":
            pre_mmwave_json = mmwave_json
        mmwave_json = json.loads(data)
        # print("mmwave:", mmwave_json["TimeStamp"])

    # pass

def show_camera(img_que, q):
    t_total = 0
    t_cnt = 0
    cap = cv2.VideoCapture(0)
    global stop_program
    while stop_program:
        ret, frame = cap.read()
        
        img_que.put(frame)
        if t_cnt%5==0:
            q.put(frame)
            
        print(img_que.qsize(), q.qsize())
        # q.put(frame)
        # cv2.imshow("Camera", frame)

        # print("camera:", datetime.now().strftime("%H:%M:%S:%f"), pre_mmwave_json["TimeStamp"], mmwave_json["TimeStamp"])
        t_err = cal_time_error(datetime.now().time(), datetime.strptime(mmwave_json['TimeStamp'], "%H:%M:%S:%f").time())

        # t_total += t_err
        t_cnt += 1
        # if t_err > 0.1:
        #     print(t_err)
        # print(t_total / t_cnt) # mean diff: 0.0478 ms, camera faster than mmwave
        


    # cap.release()
    # cv2.destroyAllWindows()

def show_img(img_que, window_name):
    global stop_program # must be declared as global var, otherwise it is a local variable in function
    while stop_program:
        # if not img_que.empty():
        frame = img_que.get()
        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(1) == ord("q"):
            stop_program = False
            break

# def show_img2(img_que):
#     global stop_program # must be declared as global var, otherwise it is a local variable in function
#     while stop_program:
#         # if not img_que.empty():
#         frame2 = img_que.get()
#         cv2.imshow("Camera2", frame2)
        
#         if cv2.waitKey(1) == ord("q"):
#             stop_program = False
#             break

if __name__ == "__main__":
    stop_program = True

    mmwave_json = pre_mmwave_json = ""
    img_que = queue.Queue()
    q = queue.Queue()

    # 建立一個UDP socket
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(("0.0.0.0", 6000))

    BUFSIZE = 2048

    # 建立一個接收sensor資料的線程
    t_mmwave = threading.Thread(target=receive_data, args=(udp_socket, BUFSIZE), daemon=True).start()
    t_display_1 = threading.Thread(target=show_img, args=(img_que,"camera1"), daemon=True).start()
    t_display_2 = threading.Thread(target=show_img, args=(q, "camera2"), daemon=True).start()

    # main
    show_camera(img_que, q)


    cv2.destroyAllWindows()


    # # 建立一個顯示相機影像的線程
    # receive_image_thread  = threading.Thread(target=show_camera)
    # receive_image_thread.start()
    # receive_image_thread.join()

