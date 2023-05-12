ir_content = """Please output the following in English, adhering strictly to the format below, about the performance of the given company in the year {year}.

format```
# Summary
A summary of announcements about performance, forecasts, and plans

# Results ({year} Sales, operating profit, ordinary income, selling, general and administrative expenses, etc. compared with the past)
(Minimum 5, maximum 10)
1. a quantitative performance (eg. sales)
2. 
3. 

# Future performance outlook (Facts and signs related to the company's future sales and profits)
(Minimum 4, maximum 6)
1. Outlook 1
2. 
3. 

# Classification keywords
(Minimum 3, maximum 4)
1. Keyword 1
2. 
3. 
```"""

ir_get_year = """"The given markdown text is the beginning of a document about a company's performance. Please output the value of 'Fiscal year of the announcement' in accordance with the following format.

Output format```
Year (e.g., 1999, 2023)
```"""

ir_header = """The given text is the cover of an existing company's performance report. Please output in English according to the output format. However, please be sure to follow the following constraints when outputting.

Constraints```
1. If the year is in 3 digits or 5 digits, estimate the correct year and output in 4 digits.
2. Only adjust the year, don't change any other information.
3. Be sure to include "# Summary" in the output.
```

Output format```
# Summary
1. Company name: Name
2. Fiscal year of the announcement: Year (eg. "2nd quarter of fiscal 2025", "Full year of March 2025")
3. Announcement date: Year/Month/Day
```"""

ir_judge_invest = """You are a professional long-term equity investor. Based on your past investment experience and knowledge, please estimate whether or not you should buy the stock of a company with the final performance. However, please output in English adhering to the following constraints.

Constraints ```
1. Make your decision based on the quantitative and qualitative data of the given performance 'results' and 'outlook'.
2. The judgement format is "positive" or "negative".
3. If the performance has been consistently improving and is likely to continue to improve, judge it as positive, otherwise judge it as negative.
4. Output the reason why you made such a judgment in bullet points.

Performance```
{performance}
```

Output format```
# The judgement
"positive" or "negative"

# The reason of your judgement
(Minimum 5, maximum 7)
1. 
2. 
3. 
4. 
5.
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