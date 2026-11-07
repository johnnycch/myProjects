import datetime

now = datetime.datetime.now()

print(now.strftime("%y-%m-%d (%A)"))
print("年份：", now.year)
print("月份：", now.month)
print("日：", now.day)
print("星期：", now.strftime("%A"))  # %A 顯示完整星期名稱
weekday_zh = ["一", "二", "三", "四", "五", "六", "日"]
print("星期：", "星期" + weekday_zh[now.weekday()])

print(now.strftime("%H:%M:%S"))
print("時間：", now.strftime("%H:%M:%S"))
print(now.strftime("%p %I:%M:%S %p"))
