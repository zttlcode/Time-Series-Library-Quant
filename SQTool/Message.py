import hmac
import hashlib
import base64
import urllib.parse
import requests
import json
import ast
import time
import smtplib
from email.utils import formataddr
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

from SQTool import Tools as SQTools


def dd(message):
    """
    钉钉提供自定义群机器人功能
    官方文档：https://open.dingtalk.com/document/group/custom-robot-access
    1、建群
    2、添加自定义机器人
    3、安全模式我选择的是加签
    4、拿到webhook地址：https://open.dingtalk.com/document/robots/customize-robot-security-settings
    5、给这个地址post报文，要发的信息放message。钉钉调用，1分钟不能超过20次，否则会禁言10分钟
    """
    # 1、当前时间戳，单位是毫秒，与请求调用时间误差不能超过1小时。
    timestamp = str(round(time.time() * 1000))
    # 2、密钥，机器人安全设置页面，加签一栏下面显示的SEC开头的字符串。
    secret = 'SEC7cae85d46265f9261e27ffedc1ddc7634bb611927b7b16af7648881c57e474a1'
    # 3、复制官方文档的签名计算代码
    secret_enc = secret.encode('utf-8')
    string_to_sign = '{}\n{}'.format(timestamp, secret)
    string_to_sign_enc = string_to_sign.encode('utf-8')
    hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
    # 4、把timestamp和第一步得到的签名值拼接到URL中
    url = 'https://oapi.dingtalk.com/robot/send?access_token=d14c065e5019197e0afe71d0f909f7aedc82855bc651e54623c00128c1447ca8&timestamp=' + timestamp + '&sign=' + sign + ''
    header = {'Content-Type': 'application/json;charset=utf-8'}
    json_text = {
        "at": {"atMobiles": [""], "atUserIds": [""], "isAtAll": False},
        "text": {
            "content": message
        },
        "msgtype": "text"
    }
    res = requests.post(url, json.dumps(json_text), headers=header, )
    # 返回的是json格式字符串，用ast转为字典类型
    resDic = ast.literal_eval(res.content.decode())
    return resDic


def wechatE(message):
    """
    企业微信提供自定义群机器人功能
    官方文档：https://developer.work.weixin.qq.com/document/path/91770
    跟钉钉步骤基本类似。企业微信要先创建团队，再添加成员，就会出现群组，然后添加机器人，返回webhook
    https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=b43f9c40-1fbe-464d-9d82-0c2a462728cd
    """
    url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=b43f9c40-1fbe-464d-9d82-0c2a462728cd'
    header = {'Content-Type': 'application/json;charset=utf-8'}
    json_text = {
        "msgtype": "text",
        "text": {
            "content": message
        }
    }
    res = requests.post(url, json.dumps(json_text), headers=header, )
    return res.ok


def telegram(message):
    """
    我这个代码是2022年12月14号写的，当时不开代理能直接调通
    2022年12月底发生白纸游行，导致telegram被墙了
    如果要用，需要pip安装代理工具，
    可以参考RMQData的bianaceAPI
    手机安装telegram的话，必须开全局代理，才能收到验证码

    官方文档：https://core.telegram.org/bots/api#authorizing-your-bot
    1、在telegram里@BotFather创建机器人并起名，会返回token
    以后调用机器人用这个地址  https://api.telegram.org/bot<token>/METHOD_NAME

    我的机器人叫 bot1 它还有个用户名叫 rbm_msg1_bot，token是 5737674853:AAGvipq6DoaBDWMqxjYT1bq5N75WHbcuIhI
    然后BotFather 返回给我一个机器人地址，https://t.me/rbm_msg1_bot，需要电脑浏览器打开，且电脑安装了telegram，打开链接会
    自动跳转telegram桌面应用，激活机器人对话框
    2、新建个channel，再点开机器人对话框，把它添加到channel里
    3、调用机器人接口更新对话列表 https://api.telegram.org/bot5737674853:AAGvipq6DoaBDWMqxjYT1bq5N75WHbcuIhI/getUpdates
    这个可能有延时，更新不到就等一会。返回报文找chat，选对话id
    4、把chat id 拿出来放进下面发信息的报文里，就可以发送信息了
    chat_id :
        5499882594  bot1机器人跟我的对话框
        -1001888987771  我新建的频道的，这个频道任何人都可以加进来
    """
    url = 'https://api.telegram.org/bot5737674853:AAGvipq6DoaBDWMqxjYT1bq5N75WHbcuIhI/sendMessage'
    header = {'Content-Type': 'application/json;charset=utf-8'}
    json_text = {
        "chat_id": "-1001888987771",
        "text": message
    }
    res = requests.post(url, json.dumps(json_text), headers=header, )
    return res.ok


def QQmail(msg, mail_list_qq):
    # 发送QQ邮件 python第三方包smtplib可以发送邮件，qq邮件能在微信里查收
    # 配置发件人邮箱
    from_addr = 'zhaot1993@qq.com'
    # 配置发件人登陆密钥
    MyPass = 'iabukotxvteujcha'
    # 配置收件人邮箱列表  sendmail的to_addrs是地址列表，可添加多个地址
    # 在配置文件里 逗号分隔 zhaot1993@qq.com,287151402@qq.com  末尾不加逗号
    to_addr = SQTools.read_config("SQT", mail_list_qq).split(",")

    ResSuccess = True

    try:
        # 配置邮件服务信息
        """  smtp不论发谁家的邮件，写法格式都一样。还能这样写
        smtp=smtplib.SMTP()
        smtp.connect('smtp.163.com')
        smtp.login(username, pwd)
        smtp.sendmail(from_addr,to_addr, msg)
        smtp.quit()
        常用邮箱SMTP服务器
        新浪邮箱：smtp.sina.com
        126邮箱：smtp.126.com
        """
        server = smtplib.SMTP_SSL('smtp.qq.com', 465)
        server.login(from_addr, MyPass)
        server.sendmail(from_addr, to_addr, msg.as_string())
        server.quit()

    except Exception as e:
        ResSuccess = False
        print(e)
    return ResSuccess


def build_msg_text_no_entity(title, post_msg):
    msg = MIMEMultipart()
    msg.attach(MIMEText(post_msg, 'html', 'utf-8'))
    # msg = MIMEText(post_msg, 'plain', 'utf-8')
    msg['From'] = formataddr(['robot', 'zhaot1993@qq.com'])
    msg['to'] = formataddr(['me', 'anonymous'])
    msg['Subject'] = title
    return msg


def build_msg_text(title, strategyResultEntity):
    # 发件人、收件人、标题、邮件内容
    # msg = build_msg_text(from_addr, to_addr, title, message)  # 发文字
    message = ('日线级别：' + strategyResultEntity.msg_level_day
               + '\n60分钟级别：' + strategyResultEntity.msg_level_60
               + '\n30分钟级别：' + strategyResultEntity.msg_level_30
               + '\n15分钟级别：' + strategyResultEntity.msg_level_15
               + '\n5分钟级别：' + strategyResultEntity.msg_level_5)
    msg = MIMEText(message, 'plain', 'utf-8')
    msg['From'] = formataddr(['QuantRobot', 'anonymous'])
    msg['to'] = formataddr(['VIP', 'anonymous'])
    msg['Subject'] = title
    return msg


def build_msg_HTML(title, strategyResultEntity):
    HTMLContent = '<html><head></head><body>' \
                    '<p>d：' + strategyResultEntity.msg_level_day +'</p>' \
                    '<p>60：' + strategyResultEntity.msg_level_60 +'</p>' \
                    '<p>30：' + strategyResultEntity.msg_level_30 +'</p>' \
                    '<p>15：' + strategyResultEntity.msg_level_15 +'</p>' \
                    '<p>5：' + strategyResultEntity.msg_level_5 +'</p>' \
                    '</body></html>'
    print(HTMLContent)
    msg = MIMEText(HTMLContent, 'html', 'utf-8')
    msg['Subject'] = title
    msg['From'] = formataddr(('robot', 'zhaot1993@qq.com'))  # 邮件上显示的发件人
    msg['To'] = formataddr(('me', 'anonymous'))  # 邮件上显示的收件人
    return msg


def build_msg_file(from_addr, to_addr, title, message, filePath, filename):
    # filePath = 'E:\\QuantData\\20221013232128.jpg'
    # filename = "20221013232128.jpg"
    # msg = build_msg_file(from_addr, to_addr, title, message, filePath, filename)  # 发文件
    HTMLContent = '<html><head></head><body>' \
                  '<h1>Hello</h1>日期'+message+'' \
                  '</body></html>'
    msg = MIMEMultipart()  # 传文件要用这个
    body = MIMEText(HTMLContent, 'html', 'utf-8')
    msg.attach(body)
    msg['Subject'] = title
    msg['From'] = formataddr(['QuantRobot', 'anonymous'])  # 邮件上显示的发件人
    msg['To'] = formataddr(['VIP', 'anonymous'])  # 邮件上显示的收件人
    # 'D:\\stockData\\ch9\\6008862019-01-012019-05-31.csv'
    file = MIMEText(open(filePath, 'rb').read(), 'plain', 'utf-8')
    file['Content-Type'] = 'application/text'
    file['Content-Disposition'] = 'attachment;filename="'+filename+'"'
    msg.attach(file)
    return msg


def build_msg_img(from_addr, to_addr, title, message, filePath, filename):
    # filePath = 'E:\\QuantData\\20221013232128.jpg'
    # filename = "20221013232128.jpg"
    # msg = build_msg_img(from_addr, to_addr, title, message, filePath, filename)
    # 发图片
    HTMLContent='<html><head></head><body><h1>Hello</h1>买点日期'+message+'' \
                '<img src="cid:' + filename + '"/></body></html>'
    # 用img标签来显示图片，其中cid是固定写法，而cid冒号后面的picAttachment需要和下面的Content - ID属性值完全一致，
    # 否则图片只能以附件的形式发送，而无法在邮件正文内以富文本的格式显示。
    msg=MIMEMultipart()
    body = MIMEText(HTMLContent, 'html', 'utf-8')
    msg.attach(body)
    msg['Subject'] = title
    msg['From'] = formataddr(['QuantRobot', 'anonymous'])   # 邮件上显示的发件人
    msg['To'] = formataddr(['VIP', 'anonymous'])  # 这个只是显示作用
    # 用MIMEImage对象来容纳本地图片
    # 'D:\\stockData\\ch10\\picAttachement.jpg'
    imageFile = MIMEImage(open(filePath, 'rb').read())
    imageFile.add_header('Content-ID', filename)
    # 要和cid冒号后面一致
    # imageFile['Content-Disposition'] = 'attachment;filename="'+filename+'.jpg"'
    imageFile['Content-Disposition'] = 'attachment;filename="' + filename + '"'
    msg.attach(imageFile)
    return msg

