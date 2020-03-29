from msvcrt import getch
import sys
import socket
my_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
my_socket.connect(("127.0.0.1", 1729))
print("connected")
data=" "
while data!=b"":
    data=my_socket.recv(1024)
    print(str(data))
my_socket.close()
print("connection terminated")

