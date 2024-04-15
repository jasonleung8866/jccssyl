import pyttsx3 # 安裝: pip install pyttsx3

speech = pyttsx3.init() # 建立文字轉語文物件

speech.setProperty('rate', 120) # 設定語速, 預設值為 200

text = "只需要簡單的幾行程式便能讓你的電腦說話"

voices = speech.getProperty('voices') # 取得可用的語音套件

# 顯示可用的語音套件 (用後可 Remark)
for i,voice in enumerate(voices):
    print(f"Voice[{i}]: {voice.name}")

speech.setProperty('voice',voices[1].id) # 根據需要哪種語音修改 voices [ ] 方括內的數值

speech.say(text)

speech.runAndWait()
