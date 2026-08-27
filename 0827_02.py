import random as r

print('업다운 게임')
print('='*50)
number = r.randint(1, 99)

chanceNum = 10
win = False

while(chanceNum >= 1):
    user = input("1에서 99 까지의 숫자를 입력하세요. > ")
    print('=' * 50)
    userNum = int(user)
    if userNum >=1 and userNum <= 99:
        if userNum < number :
            print(f"{userNum}보다 업입니다.")
            chanceNum = chanceNum - 1
            print(f"{chanceNum}번의 기회가 남았습니다.")
            print('=' * 50)

        elif userNum > number:
            print(f"{userNum}보다 다운입니다.")
            chanceNum = chanceNum - 1
            print(f"{chanceNum}번의 기회가 남았습니다.")
            print('='*50)

        elif userNum == number :
            win = True;
            break
    else:
        print("1에서 100까지의 숫자를 입력하지 못했습니다.")
        print('=' * 50)
        break

print("게임 종료")
if win:
    print("당신이 이겼습니다.")
else:
    print("당신이 졌습니다.")

