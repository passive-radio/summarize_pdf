ir_content = """与えられた企業の業績に関する文章を以下のフォーマットに必ず従い日本語で出力して下さい。```
# 要約
業績・予想・計画に関する発表の要約

# 業績のハイライト
    (最小5個、最大10個)
    1. ハイライト1
    2. 
    3. 
    

# 今後の業績見通し (入居率など)
    (最小4個、最大6個)
    1. 見通し1
    2. 
    3. 

# 分類キーワード
    (最小3個、最大4個)
    1. キーワード1
    2. 
    3. 
```"""

ir_header = """与えられた文章は実在する企業の業績レポートの表紙です。以下のフォーマットに必ず従い日本語で出力して下さい。```
# 概要
    1. 企業名
    2. 発表を行った会計年度(例: 2025年度第2四半期)
    3. 発表日
```"""

biology_lecture = """与えられた生物学に関する講義のレジュメを要約し以下のフォーマットを必ず守り日本語で出力してください。```
# Summary
    1. author
    2. institution
    3. summary

# Section1
    (more than 7 topics, more than 60 words on each topic)
    1. a highlight of this section (eg. theory, experimental result)
    2.
    3.
    4.
    5.
    6.
    7.

# Section2
    (more than 7 topics, more than 60 words on each topic)
    1. a highlight of this section (eg. theory, experimental result)
    2. 
    3. 

# Section3
    (more than 7 topics, more than 60 words on each topic)
    1. a highlight of this section (eg. theory, experimental result)
    2. 
    3. 
```"""

general = """与えられた文章を以下のフォーマットに必ず従い日本語で出力して下さい。```
# 要約
文章の要約

# ポイント1(文章の数だけ)
文章にセクション・見出しなどがあれば、その数だけポイントを立てて、そのポイント毎に要約する。

# 分類キーワード
    (最小3個、最大4個)
    1. キーワード1
    2. 
    3. 
```"""