-- ----------------------------
-- Procedure structure for PROC_XZZ_CORRECT
-- ----------------------------
CREATE OR REPLACE procedure "PROC_XZZ_CORRECT"
as  
 TYPE arry_var IS VARRAY(100) OF VARCHAR2(50); 
		disease_old_name arry_var; 
		disease_new_name arry_var; 
 
  
begin
-- 	disease_old_name:=arry_var('''510110''', '''510122''', '''510124''', '''510199''',
-- '''510626''', '''510724''', '''510771''', '''510772''', '''510776''', '''510777''', '''510778''',
-- '''511028''', '''511422''', '''511521''', '''511522''', '''511721''', '''511821''', '''512081''',
-- '''513229''', '''513321''','''四川省成都市新津县''','''四川省''','''四川''','''四川省成都''','''宣汉县东乡''','''简阳市''','''大邑县晋元''','''米易县''',
-- '''四川省阿坝州若尔盖县''','''四川省巴中市平昌县''');
--
-- 	disease_new_name:=arry_var('''510116''', '''510116''', '''510117''', '''510107''',
-- '''510604''', '''510705''', '''510703''', '''510703''', '''510703''', '''510704''', '''510704''',
-- '''511083''', '''511403''', '''511504''', '''511503''', '''511703''', '''511803''', '''510185''',
-- '''513201''', '''513301''','''510132''','''510000''','''510000''','''510100''','''511722''','''510185''','''510129''','''510421''',
-- '''513232''','''511923''');

	disease_old_name:=arry_var('''510110''', '''510122''', '''510124''', '''510199''','''510922''',
'''510626''', '''510724''', '''510771''', '''510772''', '''510776''', '''510777''', '''510778''',
'''511028''', '''511422''', '''511521''', '''511522''', '''511721''', '''511821''', '''512081''',
'''513229''', '''513321''','''510132''','''四川省成都市新津县''','''四川省''','''四川''','''四川省成都''','''宣汉县东乡''','''简阳市''','''大邑县晋元''','''米易县''',
'''四川省阿坝州若尔盖县''','''四川省巴中市平昌县''');

	disease_new_name:=arry_var('''510116''', '''510116''', '''510117''', '''510107''','''510981''',
'''510604''', '''510705''', '''510703''', '''510704''', '''510703''', '''510703''', '''510704''',
'''511083''', '''511403''', '''511504''', '''511503''', '''511703''', '''511803''', '''510185''',
'''513201''', '''513301''','''510118''','''510118''','''510000''','''510000''','''510100''','''511722''','''510185''','''510129''','''510421''',
'''513232''','''511923''');

		
		for i in 1..disease_old_name.count loop--修改列
				dbms_output.put_line(i);
				execute immediate 'update BA_SC set DEPT_ADRRESSCODE2='||disease_new_name(i)||' where DEPT_ADRRESSCODE2='||disease_old_name(i);
				commit;
				
				execute immediate 'update BA_SC set XZZ_XZQH2='||disease_new_name(i)||' where XZZ_XZQH2='||disease_old_name(i);
				commit;
				
		end loop;
end;