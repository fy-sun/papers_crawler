# 环境安装
```python
pip install requests beautifulsoup4 pandas tqdm tenacity
```

# 使用

**现有三个研究方面的正则项：多模态、分割、多模态&分割**

**直接替换MULTIMODAL_REGEX(MM_RE)即可**

> **注：当前ICML和ICLR在2024及以前不可用，ID获取方式需要更改；NeurIPS2025尚未公开。**

## CVPR & ICCV
```python
# 执行cvpr_scraper.py/iccv_scraper.py获取详细信息
python cvpr_scraper.py --year 2025 --outdir ./cvpr2025

# 启动下载程序
python cvpr_download.py --csv ./files/cvpr2025_full.csv --outdir ./downloads --year 2025 --conf CVPR
```

## ICLR & ICML & NeurIPS
```python
# 首先下载csv文件
# 启动下载程序
python nips_download.py --csv ./files/NeurIPS 2024 Events.csv --outdir ./downloads --year 2024 --conf NeurIPS
```

# 论文信息下载/搜索地址

- CVPR
https://openaccess.thecvf.com/CVPR{year}?day=all

- ICCV
https://openaccess.thecvf.com/ICCV{year}?day=all

- NeurIPS
https://neurips.cc/Downloads/{year}

- ICML
https://icml.cc/Downloads/{year}

- ICLR
https://iclr.cc/Downloads/{year}
