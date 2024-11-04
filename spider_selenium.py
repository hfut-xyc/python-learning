from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
import pandas as pd
import time

url = 'https://data.eastmoney.com/bbsj/201806/yjbb.html'

driver = webdriver.Chrome()
driver.implicitly_wait(10)
driver.get(url)
wait = WebDriverWait(driver, 10)

page = 3
input = driver.find_element(By.ID, 'gotopageindex')
input.click()
input.clear()
input.send_keys(page)

goto = driver.find_element(By.XPATH, "//input[@value='Go']")
goto.submit()
# time.sleep(2)
wait.until(EC.text_to_be_present_in_element(
    (By.XPATH, '//a[@class="active"]'), str(page)))

# thead_list = driver.find_elements(By.XPATH, "//div[@class='dataview-body']//thead/tr[1]//th")
# thead_list = [item.text.replace('\n', '') for item in thead_list]
# print(thead_list)

data_list = []
for i in range(1, 50 + 1):
    row = driver.find_elements(
        By.XPATH, f"//div[@class='dataview-body']//tbody/tr[{i}]/td")
    # print(row[0])
    row = [column.text for column in row]
    data_list.append(row)

print(*data_list[:10], sep='\n')
