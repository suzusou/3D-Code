# 3D-Code

##  アプリ概要
- 色情報と文字情報を組み合わせた３次元コード
- ２次元コードよりも情報の種類を増やしたため、情報格納量が増加  
- 誤り訂正は、QRコードにも採用されているリードソロモン符号

## URL
[3D-Code App](https://yax.f5.si)

## 作成メンバー
[Bellfat](https://github.com/suzusou) , [Surugawann](https://github.com/Surugawann) , [シマ](https://github.com/marz0723) , [yamashun778899](https://github.com/yamashun778899)

## 使用技術
### Server
- Docker (APIサーバー)
- nginx (Webサーバー)

### Python3.11.7
- openCV (画像処理)
- PyOCR (文字認識)
- Reedsolo (誤り訂正)

## 技術説明
[こちらのPDFをご覧ください](3D-Code.pdf)

## 強み
- 暗号化されているので外部からは読みとられない
- QRコードより同じ面積当たりの情報格納量が多い

## 問題点
- 外部の影響を受けやすい (外部光、背景など)
- 現状、色検知、文字認識の精度により、3D-Codeを読み取るときはかなり綺麗に読みとらないといけない

