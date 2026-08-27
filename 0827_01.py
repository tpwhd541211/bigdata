import random as r

print('가위 바위 보 게임')
print('-'*50)

com = r.choice(['가위', '바위', '보'])
choi = ['가위', '바위', '보']

user = input("가위 바위 보 중에 하나를 입력하세요.> ")

if user in choi:
    print(f"컴퓨터는 {com} 를 냈습니다.")
    print(f"당신은 {user} 를 냈습니다.")

    if user == "가위" and com == "보" or \
            user == "바위" and com == "가위" or \
            user == "보" and com == "바위":
        print("당신이 이겼습니다.")
    else:
        print("당신이 졌습니다.")

else :
    print("잘못된 입력입니다.")