# code for data scrapping 



from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import time
import pandas as pd
driver=webdriver.Chrome()
data1=[]
data2=[]
for i in range(1,3):
    j=372 if i==1 else 132
    for k in range(0,j+1,12):  
        driver.get(f'https://www.shl.com/solutions/products/product-catalog/?start={k}&type={i}&type={i}')
        time.sleep(3)
        elems=driver.find_elements(By.CLASS_NAME,"custom__table-responsive")
        for elem in elems:
            table_div=elem
            table=table_div.find_element(By.TAG_NAME,"table")

            rows=table.find_elements(By.TAG_NAME,"tr")
            cols_th=rows[0].find_elements(By.TAG_NAME,"th")

            for row in rows:
                cols=row.find_elements(By.TAG_NAME,"td")
                if(cols):
                    link_elem=cols[0].find_element(By.TAG_NAME,"a")
                    title=link_elem.text
                    href=link_elem.get_attribute("href")

                    if cols[1].find_elements(By.CSS_SELECTOR, "span.catalogue__circle.-yes"):
                        remote_testing=1
                    else:
                        remote_testing=0    

                    if cols[2].find_elements(By.CSS_SELECTOR, "span.catalogue__circle.-yes"):
                        adaptive_irt=1
                    else:
                        adaptive_irt=0 

                    test_types = cols[3].text.replace('\n',' ')
                    
                    if cols_th[0].text.strip() == 'Pre-packaged Job Solutions':
                        data1.append({
                            'title':title,
                            'link':href,
                            'Remote Testing':remote_testing,
                            'Adaptive_irt':adaptive_irt,
                            'test_types':test_types
                                        })
                        
                    else:
                        data2.append({
                            'title':title,
                            'link':href,
                            'Remote Testing':remote_testing,
                            'Adaptive_irt':adaptive_irt,
                            'test_types':test_types
                                                     })
                    

                   
df1=pd.DataFrame(data1)
df2=pd.DataFrame(data2)

df1.to_csv("shl_pre_packaged.csv",index=False)
df2.to_csv("shl_individual.csv",index=False)



