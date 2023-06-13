import datetime
import os
from tqdm import tqdm

def calculate_movement(price_dict_, span_, DATA_DIR):
    min_per = -100
    errorcompany = ['SCANA Corp', 'General Growth Properties Inc.']
    stock_movement_3days = []
    text_all = []
    text_titles = os.listdir(DATA_DIR)

    test_text_all = []
    test_stock_movement_3days = []

    date = []
    company_ = []
    iter_ = 0
    for t in tqdm(text_titles):
        iter_ += 1
        cur_dir = os.path.join(DATA_DIR, t)
        cur_dir = cur_dir + '/Text.txt'
        with open(cur_dir, 'r',encoding='utf-8') as f:
            text = f.read().strip().replace("<br />", " ")

        date_ = datetime.datetime.strptime(t[-8:], '%Y%m%d').strftime('%Y-%m-%d')


        company = t[:-9]
        if company in errorcompany:
            continue
        else:
            stock_price = price_dict_[company]['Adj Close']
            try:
                today_index = stock_price.index.get_loc(date_)
            except KeyError:
                try:
                    today_index = stock_price.index.get_loc(
                        (datetime.datetime.strptime(date_, '%Y-%m-%d') - datetime.timedelta(1)).strftime('%Y-%m-%d'))
                except KeyError:
                    try:
                        today_index = stock_price.index.get_loc(
                            (datetime.datetime.strptime(date_, '%Y-%m-%d') - datetime.timedelta(2)).strftime(
                                '%Y-%m-%d'))
                    except KeyError:
                        try:
                            today_index = stock_price.index.get_loc(
                                (datetime.datetime.strptime(date_, '%Y-%m-%d') - datetime.timedelta(3)).strftime(
                                    '%Y-%m-%d'))
                        except:
                            print('Error Company: ' + str(company) + 'Date: ' + date_)
                            pass
                            continue

            today_data = stock_price.iloc[today_index]
            try:
                following_nday_price = stock_price.iloc[stock_price.index.get_loc(
                        (stock_price.index[today_index] + datetime.timedelta(span_)).strftime('%Y-%m-%d'))]
            except KeyError:
                try:
                    following_nday_price = stock_price.iloc[stock_price.index.get_loc(
                        (stock_price.index[today_index] + datetime.timedelta(span_+1)).strftime('%Y-%m-%d'))]
                except KeyError:
                    try:
                        following_nday_price = stock_price.iloc[stock_price.index.get_loc(
                            (stock_price.index[today_index] + datetime.timedelta(span_+2)).strftime('%Y-%m-%d'))]
                    except KeyError:
                        try:
                            following_nday_price = stock_price.iloc[stock_price.index.get_loc(
                                (stock_price.index[today_index] + datetime.timedelta(span_+3)).strftime('%Y-%m-%d'))]
                        except:
                            print('Error following date: ' + str(company) + 'Date: ' + date_)
                            pass
                            continue

            # print(len(following_nday_data))
            if following_nday_price and today_data:
                movement = float(following_nday_price - today_data)
                persent = movement/today_data*100
                if 104.28465722801789  > persent > min_per:
                    min_per = persent
                # print(str(persent)+"%")
                if persent == 80.39968485780167:
                    print(date_,str(company))
                if movement > 0:
                    movement = 1.0
                else:
                    movement = 0.0
            else:
                print("Date Error, following_nday_price or today_data is null")
                continue
            if text == "":
                print("Error File: no content", t)
                continue
            text_all.append(text)
            stock_movement_3days.append(movement)
            date.append(date_)
            company_.append(str(company))
            # if iter_ > 450:
            #     print('Company: ' + str(company) + 'Date: ' + date_)
            #     test_text_all.append(text)
            #     test_stock_movement_3days.append(movement)
    # print(len(test_text_all), len(test_stock_movement_3days))
    print(min_per)
    return stock_movement_3days, text_all, date, company_

