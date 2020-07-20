import requests
import pymysql

NUM_OF_STOCK = 10
INIT_ID = 603227

def store_data(stock_info):
    # create connection with mysql database and do insert operation
    conn = pymysql.connect(host='localhost',user='root',password="yuhaofei",port=3306,database="xueqiu_spider",charset="utf8")
    cursor = conn.cursor()
    SQLsentence = "insert into stock values(\"%s\",%f,%f,%f)"%(stock_info[0],stock_info[1],stock_info[2],stock_info[3])
    query = cursor.execute(SQLsentence)
    conn.commit()

def main():
    # define header to pretend being a wet browser
    headers = {'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'}
    # log in to automatically get the cookie and continue crawling
    # if not use session to get cookie, crawling would be fouund
    session = requests.session()
    session.get(url="https://xueqiu.com",headers=headers)

    iter = 0;
    id = INIT_ID
    while (iter<NUM_OF_STOCK):
        # since xueqiu.com use ajax to load web page
        # we catch  XHR object to get json
        url = 'https://stock.xueqiu.com/v5/stock/quote.json?symbol=SH'+str(id).zfill(6)
        try:
            json_data = session.get(url,headers=headers).json()
            stock_info = []
            stock_info.append(str(json_data['data']['quote']['name']))
            stock_info.append(float(json_data['data']['quote']['current']))
            stock_info.append(float(json_data['data']['quote']['chg']))
            stock_info.append(float(json_data['data']['quote']['percent']))
            print('Current Stock Name: ' + stock_info[0])
            store_data(stock_info)
            id += 1
            iter += 1
        except:
            id += 1

if __name__ == '__main__':
    main()