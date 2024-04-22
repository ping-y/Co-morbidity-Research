-- 备份一个BA_SC
create table BA_SC_backup as select * from BA_SC;


--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
-- 处理地址和转码
--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
alter table BA_SC add DEPT_ADRRESSCODE2 varchar2(6);
alter table BA_SC add XZZ_XZQH2 varchar2(254);--新增列
update BA_SC set DEPT_ADRRESSCODE2=DEPT_ADRRESSCODE;

update BA_SC set XZZ_XZQH2=XZZ_XZQH;--赋值

-- 处理地址和转码
call Proc_xzz_correct();

--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
--新增字段和字段清洗 更新不符合规范的字段
--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
--第一阶段
--对疾病谱的条件进行筛选
COMMIT;
SAVEPOINT A;
CREATE INDEX IDX_BA_SC_SFZH ON BA_SC(SFZH);
SELECT COUNT(SFZH),COUNT(DISTINCT SFZH) FROM BA_SC;  --记录运行之前的人次和人数
DROP INDEX IDX_BA_SC_SFZH;
--删除疾病编码不符合规范的
CREATE INDEX IDX_BA_SC_JBDM ON BA_SC(JBDM);
UPDATE BA_SC SET JBDM=UPPER(SUBSTR(JBDM,1,1))||SUBSTR(JBDM,2)
		WHERE SUBSTR(JBDM,1,1) BETWEEN 'a' AND 'z';
DELETE FROM BA_SC WHERE SUBSTR(JBDM,1,3) BETWEEN 'U04' AND 'U89'
				OR SUBSTR(JBDM,1,3) BETWEEN 'V01' AND 'Y98';
DROP INDEX IDX_BA_SC_JBDM;

--删除不是四川的
CREATE INDEX IDX_BA_SC_XZZ_XZQH2 ON BA_SC(XZZ_XZQH2);
DELETE FROM BA_SC WHERE SUBSTR(XZZ_XZQH2,1,2)!='51' OR  XZZ_XZQH2 IS NULL;
DROP INDEX IDX_BA_SC_XZZ_XZQH2;

--删除不是二级以上医院的
CREATE INDEX IDX_BA_SC_YYDJ_J ON BA_SC(YYDJ_J);
DELETE FROM BA_SC WHERE (YYDJ_J!='2' AND  YYDJ_J!='3') OR  YYDJ_J IS NULL;
DROP INDEX IDX_BA_SC_YYDJ_J;
COMMIT;
--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
--第二阶段 所有疾病诊断的首字母大写
SAVEPOINT B;

CREATE INDEX IDX_BA_SC_JBDM1 ON BA_SC(JBDM1);
UPDATE BA_SC SET JBDM1=UPPER(SUBSTR(JBDM1,1,1))||SUBSTR(JBDM1,2)
    WHERE SUBSTR(JBDM1,1,1) BETWEEN 'a' AND 'z';
DROP INDEX IDX_BA_SC_JBDM1;

CREATE INDEX IDX_BA_SC_JBDM2 ON BA_SC(JBDM2);
UPDATE BA_SC SET JBDM2=UPPER(SUBSTR(JBDM2,1,1))||SUBSTR(JBDM2,2)
    WHERE SUBSTR(JBDM2,1,1) BETWEEN 'a' AND 'z';
DROP INDEX IDX_BA_SC_JBDM2;

CREATE INDEX IDX_BA_SC_JBDM3 ON BA_SC(JBDM3);
UPDATE BA_SC SET JBDM3=UPPER(SUBSTR(JBDM3,1,1))||SUBSTR(JBDM3,2)
    WHERE SUBSTR(JBDM3,1,1) BETWEEN 'a' AND 'z';
DROP INDEX IDX_BA_SC_JBDM3;

CREATE INDEX IDX_BA_SC_JBDM4 ON BA_SC(JBDM4);
UPDATE BA_SC SET JBDM4=UPPER(SUBSTR(JBDM4,1,1))||SUBSTR(JBDM4,2)
    WHERE SUBSTR(JBDM4,1,1) BETWEEN 'a' AND 'z';
DROP INDEX IDX_BA_SC_JBDM4;

CREATE INDEX IDX_BA_SC_JBDM5 ON BA_SC(JBDM5);
UPDATE BA_SC SET JBDM5=UPPER(SUBSTR(JBDM5,1,1))||SUBSTR(JBDM5,2)
    WHERE SUBSTR(JBDM5,1,1) BETWEEN 'a' AND 'z';
DROP INDEX IDX_BA_SC_JBDM5;

CREATE INDEX IDX_BA_SC_JBDM6 ON BA_SC(JBDM6);
UPDATE BA_SC SET JBDM6=UPPER(SUBSTR(JBDM6,1,1))||SUBSTR(JBDM6,2)
    WHERE SUBSTR(JBDM6,1,1) BETWEEN 'a' AND 'z';
DROP INDEX IDX_BA_SC_JBDM6;

CREATE INDEX IDX_BA_SC_JBDM7 ON BA_SC(JBDM7);
UPDATE BA_SC SET JBDM7=UPPER(SUBSTR(JBDM7,1,1))||SUBSTR(JBDM7,2)
    WHERE SUBSTR(JBDM7,1,1) BETWEEN 'a' AND 'z';
DROP INDEX IDX_BA_SC_JBDM7;

CREATE INDEX IDX_BA_SC_JBDM8 ON BA_SC(JBDM8);
UPDATE BA_SC SET JBDM8=UPPER(SUBSTR(JBDM8,1,1))||SUBSTR(JBDM8,2)
    WHERE SUBSTR(JBDM8,1,1) BETWEEN 'a' AND 'z';
DROP INDEX IDX_BA_SC_JBDM8;

CREATE INDEX IDX_BA_SC_JBDM9 ON BA_SC(JBDM9);
UPDATE BA_SC SET JBDM9=UPPER(SUBSTR(JBDM9,1,1))||SUBSTR(JBDM9,2)
    WHERE SUBSTR(JBDM9,1,1) BETWEEN 'a' AND 'z';
DROP INDEX IDX_BA_SC_JBDM;

CREATE INDEX IDX_BA_SC_JBDM10 ON BA_SC(JBDM10);
UPDATE BA_SC SET JBDM10=UPPER(SUBSTR(JBDM10,1,1))||SUBSTR(JBDM10,2)
    WHERE SUBSTR(JBDM10,1,1) BETWEEN 'a' AND 'z';
DROP INDEX IDX_BA_SC_JBDM10;

CREATE INDEX IDX_BA_SC_JBDM11 ON BA_SC(JBDM11);
UPDATE BA_SC SET JBDM11=UPPER(SUBSTR(JBDM11,1,1))||SUBSTR(JBDM11,2)
    WHERE SUBSTR(JBDM11,1,1) BETWEEN 'a' AND 'z';
DROP INDEX IDX_BA_SC_JBDM11;

CREATE INDEX IDX_BA_SC_JBDM12 ON BA_SC(JBDM12)
UPDATE BA_SC SET JBDM12=UPPER(SUBSTR(JBDM12,1,1))||SUBSTR(JBDM12,2)
    WHERE SUBSTR(JBDM12,1,1) BETWEEN 'a' AND 'z';
DROP INDEX IDX_BA_SC_JBDM12;

CREATE INDEX IDX_BA_SC_JBDM13 ON BA_SC(JBDM13);
UPDATE BA_SC SET JBDM13=UPPER(SUBSTR(JBDM13,1,1))||SUBSTR(JBDM13,2)
    WHERE SUBSTR(JBDM13,1,1) BETWEEN 'a' AND 'z';
DROP INDEX IDX_BA_SC_JBDM13;;

CREATE INDEX IDX_BA_SC_JBDM14 ON BA_SC(JBDM14);
UPDATE BA_SC SET JBDM14=UPPER(SUBSTR(JBDM14,1,1))||SUBSTR(JBDM14,2)
    WHERE SUBSTR(JBDM14,1,1) BETWEEN 'a' AND 'z';
DROP INDEX IDX_BA_SC_JBDM14;

CREATE INDEX IDX_BA_SC_JBDM15 ON BA_SC(JBDM15);
UPDATE BA_SC SET JBDM15=UPPER(SUBSTR(JBDM15,1,1))||SUBSTR(JBDM15,2)
    WHERE SUBSTR(JBDM15,1,1) BETWEEN 'a' AND 'z';
DROP INDEX IDX_BA_SC_JBDM15;

--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
--第三阶段 根据字段实际异常值情况，清洗字段
-- 更新一些字段。需要查询具体字段的内容再更新
SAVEPOINT C;

-- LSGX
CREATE INDEX IDX_BA_SC_LSGX ON BA_SC(LSGX);
SELECT LSGX, COUNT(*) FROM BA_SC GROUP BY LSGX;
UPDATE BA_SC SET LSGX = 99 WHERE LSGX IS NULL;
DROP INDEX IDX_BA_SC_LSGX;

--3.YYDJ_J
CREATE INDEX IDX_BA_SC_YYDJ_J ON BA_SC(YYDJ_J);
SELECT YYDJ_J, COUNT(*) FROM BA_SC GROUP BY YYDJ_J;
UPDATE BA_SC SET YYDJ_J = 9 WHERE YYDJ_J IS NULL;
DROP INDEX IDX_BA_SC_YYDJ_J;

--4.YYDJ_D
CREATE INDEX IDX_BA_SC_YYDJ_D ON BA_SC(YYDJ_D);
SELECT YYDJ_D, COUNT(*) FROM BA_SC GROUP BY YYDJ_D;
UPDATE BA_SC SET YYDJ_D = 9 WHERE YYDJ_D IS NULL;
DROP INDEX IDX_BA_SC_YYDJ_D;

--5.YLFKFS
CREATE INDEX IDX_BA_SC_YLFKFS ON BA_SC(YLFKFS);
SELECT YLFKFS, COUNT(*) FROM BA_SC GROUP BY YLFKFS;
UPDATE BA_SC SET YLFKFS = 99 WHERE YLFKFS = '09';
UPDATE BA_SC SET YLFKFS = 99 WHERE YLFKFS = '-';
UPDATE BA_SC SET YLFKFS = 99 WHERE YLFKFS IS NULL;
DROP INDEX IDX_BA_SC_YLFKFS;

--6.XB
CREATE INDEX IDX_BA_SC_XB ON BA_SC(XB);
SELECT XB, COUNT(*) FROM BA_SC GROUP BY XB;
UPDATE BA_SC SET XB = 9 WHERE XB = '-';
UPDATE BA_SC SET XB = 9 WHERE XB IS NULL;
UPDATE BA_SC SET XB = 9 WHERE XB = '0';
UPDATE BA_SC SET XB = 1 WHERE XB = '1.0';
UPDATE BA_SC SET XB = 2 WHERE XB = '2.0';
DROP INDEX IDX_BA_SC_XB;

--7.ZY
CREATE INDEX IDX_BA_SC_ZY ON BA_SC(ZY);
SELECT ZY, COUNT(*) FROM BA_SC GROUP BY ZY;
UPDATE BA_SC SET ZY = 90 WHERE ZY = '-';
UPDATE BA_SC SET ZY = 90 WHERE ZY IS NULL;
UPDATE BA_SC SET ZY = 17 WHERE ZY = '17.0';
DROP INDEX IDX_BA_SC_ZY;

--10.MZ
CREATE INDEX IDX_BA_SC_MZ ON BA_SC(MZ);
SELECT MZ, COUNT(*) FROM BA_SC GROUP BY MZ;
UPDATE BA_SC SET MZ = 99 WHERE MZ IS NULL;
UPDATE BA_SC SET MZ = 99 WHERE MZ = '-';
DROP INDEX IDX_BA_SC_MZ;

--HY 10 未婚 ，20 已婚，30 离异丧偶， 90 缺失
CREATE INDEX IDX_BA_SC_HY ON BA_SC(HY);
SELECT HY, COUNT(*) FROM BA_SC GROUP BY HY;
UPDATE BA_SC SET HY = 90 WHERE HY = '-';
UPDATE BA_SC SET HY = 90 WHERE HY = '9';
UPDATE BA_SC SET HY = 90 WHERE HY = '0';
UPDATE BA_SC SET HY = 90 WHERE HY IS NULL;
UPDATE BA_SC SET HY = 10 WHERE HY = '1';
UPDATE BA_SC SET HY = 20 WHERE HY = '2';
UPDATE BA_SC SET HY = 30 WHERE HY = '3';
UPDATE BA_SC SET HY = 40 WHERE HY = '4';
UPDATE BA_SC SET HY = 20 WHERE HY = '21' OR HY = '22' OR HY = '23';
UPDATE BA_SC SET HY = 30 WHERE HY = '40';
UPDATE BA_SC SET HY = 10 WHERE HY = '90' AND XB = '1' AND NL < '22';
UPDATE BA_SC SET HY = 10 WHERE HY = '90' AND XB = '2' AND NL < '20';
DROP INDEX IDX_BA_SC_HY;

--处理时间异常值
CREATE INDEX IDX_BA_SC_CSRQ ON BA_SC(CSRQ);
UPDATE BA_SC SET CSRQ=NULL WHERE LENGTH(CSRQ)!=8 AND LENGTH(CSRQ)!=10;
DROP INDEX IDX_BA_SC_CSRQ;
UPDATE BA_SC SET RYSJ=NULL WHERE LENGTH(RYSJ)!=8 AND LENGTH(RYSJ)!=10;
UPDATE BA_SC SET CYSJ=NULL WHERE LENGTH(CYSJ)!=8 AND LENGTH(CYSJ)!=10;

--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
-- --新增贫困县和摘帽时间的字段。
-- --新增两个字段 贫困县认定、贫困县摘帽时间  varchar2（10）
-- alter table BA_SC add 贫困县认定 VARCHAR2(10);
-- alter table BA_SC add 贫困县摘帽时间 VARCHAR2(10);
--
-- update BA_SC set 贫困县认定=(select poor from aa_poor_city
-- 				where BA_SC.XZZ_XZQH2=aa_poor_city.XZZ_XZQH2); --应该可以优化 改成EXISTS
-- update BA_SC set 贫困县认定='0' where 贫困县认定 is NULL;
--
-- update BA_SC set 贫困县摘帽时间=(select year from aa_poor_city
-- 				where BA_SC.XZZ_XZQH2=aa_poor_city.XZZ_XZQH2); --应该可以优化 改成EXISTS
-- update BA_SC set 贫困县摘帽时间='0' where 贫困县摘帽时间 is NULL;

-- --添加经济区字段
-- alter table BA_SC add economic_area VARCHAR2(10);
-- update BA_SC set economic_area=(select economic from economic_areas where code4=substr(XZZ_XZQH2,1,4)); --应该可以优化 改成EXISTS

--mz 2 为varchar  不能number
alter table BA_SC  add (MZ_2 varchar2(10)); --新增字段，汉族1，少数民族2，缺失9
update BA_SC set MZ_2 = MZ;
CREATE INDEX IDX_BA_SC_MZ_2 ON BA_SC(MZ_2);
update BA_SC set MZ_2 = 1 where MZ_2 ='01';
update BA_SC set MZ_2 = 2 where MZ_2>'01' and MZ_2<'57' and MZ_2!='1';
DROP INDEX IDX_BA_SC_MZ_2;

--添加all desease
alter table BA_SC add (all_disease varchar2(180));----新增字段
UPDATE BA_SC SET all_disease= substr(JBDM,1,3) ||','||substr(JBDM1,1,3) ||','||substr(JBDM2,1,3) ||','||substr(JBDM3,1,3) ||','||
substr(JBDM4,1,3) ||','||substr(JBDM5,1,3) ||','||substr(JBDM6,1,3) ||','||substr(JBDM7,1,3) ||','||substr(JBDM8,1,3) ||','||
substr(JBDM9,1,3) ||','||substr(JBDM10,1,3) ||','||substr(JBDM11,1,3) ||','||substr(JBDM12,1,3) ||','||substr(JBDM13,1,3) ||','||
substr(JBDM14,1,3) ||','||substr(JBDM15,1,3);------用逗号将诊断连接起来

--添加 RY_DATE 、 CY_DATE 、 CY_YEAR 、 CS_DATE
alter table BA_SC add (RY_DATE DATE);
alter table BA_SC add (CY_DATE DATE);
alter table BA_SC add (CY_YEAR VARCHAR2(8));
alter table BA_SC add (CS_DATE DATE);


-- 先检查确定RYSJ、CYSJ、CSRQ的字符串格式
UPDATE BA_SC SET RY_DATE=TO_DATE(RYSJ,'yyyy-mm-dd');
UPDATE BA_SC SET CY_DATE=TO_DATE(CYSJ,'yyyy-mm-dd');
UPDATE BA_SC SET CY_YEAR=SUBSTR(CYSJ,1,4);
UPDATE BA_SC SET CS_DATE=TO_DATE(CSRQ,'yyyy-mm-dd');
COMMIT;

--添加flags字段
alter table BA_SC add FLAGS varchar2(6) DEFAULT '6';

CREATE INDEX IDX_BA_SC_SFZH ON BA_SC(SFZH);
update BA_SC set FLAGS='1' where length(SFZH) !=32;
update BA_SC set FLAGS='1' where SFZH  is null;
update BA_SC set FLAGS='5' where SFZH in (select SFZH from
(select SFZH, CY_YEAR ,count(0) has from BA_SC
group by SFZH,CY_YEAR) where has>30); --应该可以优化 写成EXISTS

update BA_SC set FLAGS='2' where sfzh in (select SFZH from
(select distinct SFZH,XB  from BA_SC ) group by SFZH having count(*) >1); -- IN 改写成EXISTS

update BA_SC set FLAGS='3' where sfzh in (select SFZH from
(select distinct SFZH,CS_DATE  from BA_SC )
group by SFZH having count(*) >1); -- IN 改写成EXISTS

update BA_SC set FLAGS='3' where sfzh in (select DISTINCT sfzh from BA_SC
where Abs(floor((ry_date-cs_date)/365)-nl)>2); -- IN 改写成EXISTS

update BA_SC set FLAGS='4' where SFZH in
(select distinct SFZH from BA_SC where lyfs='5'); -- IN 改写成EXISTS

-- update BA_SC set FLAGS='6' where FLAGS is NULL;

DROP INDEX IDX_BA_SC_SFZH;

-- --添加RN字段
create table BA_SC_2015to2022 as
select
		t.*,
    ROW_NUMBER() OVER (PARTITION BY SFZH ORDER BY CYSJ) as RN
from BA_SC t;






