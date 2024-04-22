pdir='E:/UESTC_yang/0611_data/'

celebro=['I6'+str(i) for i in range(10)]

SFZH,RN,MZ,XB,YYDJ_J,YYDJ_D,YLFKFS,NL,CS_DATE,ZY,HY,RYTJ,RY_DATE,CY_DATE,RYQK, LYFS, \
ZFY, ZFJE, ZB1, ZB2, ZB3, ZB4, ZB5, ZB6, ZB7, ZB8, ZB9, QTF, \
DEPT_ADDRESSCODE2, XZZ_XZQH2, FLAGS,ALL_DISEASE=[
    'SFZH','CY_DATE','MZ','XB','YYDJ_J','YYDJ_D','YLFKFS','NL','CS_DATE','ZY','HY','RYTJ', 'RY_DATE','CY_DATE','RYQK','LYFS',
    'ZFY', 'ZFJE', 'ZB1', 'ZB2', 'ZB3', 'ZB4', 'ZB5', 'ZB6', 'ZB7', 'ZB8', 'ZB9', 'QTF',
            'DEPT_ADRRESSCODE2', 'XZZ_XZQH2', 'FLAGS','ALL_DISEASE']

# celebro_category=dict(zip([i for i in range(7)],[['I60'],['I61','I62'],['I63'],['I64'],['I65','I66'],['I67','I68'],['I69']]))
celebro_category={0: ['I60'], 1: ['I61', 'I62'], 2: ['I63'], 3: ['I64'], 4: ['I65', 'I66'], 5: ['I67', 'I68'], 6: ['I69']}
category_name={0: '蛛网膜下出血', 1: '出血性卒中', 2: '缺血性卒中', 3:'未特指脑卒中', 4:'动脉闭塞和狭窄', 5:'其他脑血管病', 6: '脑血管后遗症'}

YLJGID='YLJGID'

