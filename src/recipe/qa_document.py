ir_content = """与えられた企業の2022年度の業績に関する文章を以下のフォーマットに必ず従い日本語で出力して下さい。```
# 要約
業績・予想・計画に関する発表の要約

# 結果 (2022年度の売上高、営業利益、経常利益、販管費などの数値と過去との比較)
    (最小5個、最大10個)
    1. ハイライト1
    2. 
    3. 
    

# 今後の業績見通し (企業の将来の売上・利益に関わる事実・兆し)
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

ir_header = """与えられた文章は実在する企業の業績レポートの表紙です。出力フォーマットに必ず従い日本語で出力して下さい。ただし以下の制約を必ず守って出力すること。

制約```
1. 西暦が3桁や5桁になる場合、正しい年を推定して4桁で出力する
2. 年を修正するだけで他の情報を改変しない
3. "# 概要" も必ず出力する
```

出力フォーマット```
# 概要
1. 企業名
2. 発表を行った会計年度 (eg. "2025年度第2四半期", "2025年3月期通期")
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