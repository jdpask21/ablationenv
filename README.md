# Ablationとテストケース学習に基づく実行情報の欠落を利用した故障箇所特定

## 各ディレクトリとファイルの中身について
- fl
<br>ドッカーファイルが入っています。
- workspace
<br>
    - bin<br>
        実験で使用するプログラムや実験データなどが入っています。
        - ablationex_util<br>
            実験で使用するプログラムで使われる共通の関数などが入っています。
        - images-susscore<br>
            本研究と比較手法を実験対象プログラムに適用した結果が入っています。
        - line_sfl_data<br>
            実験結果が入っています。
        - seq_sus_{ab,ochiai,tara}<br>
            各プログラムの各行に対する疑わしさスコアのデータが入っています。実験結果となるデータです。
    - chunks<br>
        各プログラムのバグの行番号を示すチャンクが入っています。
    - clover-line<br>
        本研究で使用した実行トレースが入っています。桐生さんの引継ぎ資料を参照することで同じものが収集可能です。

## /workspace/binのプログラムについて
重要でないプログラムも残していますが、説明は省略しています。
- cal_ochiai.py<br>
    OchiaiやTarantulaを実行するためのプログラムです。
- cal_bugpattern.py<br>
    実験結果を故障別に考察するために使用したプログラムです。使わないと思いますが残しています。
- checkout.sh, compile.sh, run_checkout.sh, run_test.sh, setup.sh, zoltar.sh, collect_coverage_parallelized.py, hoge.py, hoge2.py, coverageout.py, writemavenxml.py, xml_write_test.py, SIngleJUnitTestRUnner.class<br>
    実行トレースやカバレッジを収集するためのシェルとプログラムです。詳しくは[引継ぎ資料](./workspace/bin/readme_kiryu.md)を確認してください。
    <!-- 実行トレースやカバレッジを収集するためのシェルとプログラムです。OpenCloverを使用しています。 -->
- collect_bug_multi.txt, collect{PROJECT_NAME}.txt<br>
    多重故障のプロジェクトと各プロジェクトの名前とVersionが格納されたテキストファイルです。
- main_line.py<br>
    本論文における、DNNモデルの学習と保存を行うプログラムです。デフォルトでは全てのデータを訓練データにするように設定されています。使用方法などは後述。
- onlin_line.py<br>
    本研究で提案する手法を用いて疑わしさスコアを算出するプログラムです。使用にはmain_line.pyで作成したDNNモデルが必要。使用方法は後述。
- plot_susscore.py<br>
    onlin_line.pyで出力した疑わしさスコアをプロットするためのプログラムです。考察などのために使用しました。
- rename_model.py<br>
    複数のモデルを用いてその平均を最終的な実験結果とする際、保存したDNNモデルの名前を統一したほうが実験がやりやすいです。このプログラムはmain_line.pyで保存したモデルの名前をonlin_line.pyでまとめて処理できるように変更するプログラムです。
- make_linetr.py<br>
    gcovを使用して収集したカバレッジレポートから本研究で使用した実行トレースを収集するためのプログラムです。SIR用に作成したものなので、Defects4jのカバレッジレポートには使えません。Defects4jのほうはxml形式なのでプログラム書けば同じものが収集できるはずです。

## 実験環境の構築とプログラムの実行
1. READMEがあるディレクトリでDockerイメージの作成、コンテナ起動を行う。
```shell:READMEがあるディレクトリ上
docker-compose up -d
```
2. 起動したコンテナに入り、/workspace/bin/に移動
```shell:
docker exec -it [CONTAINER_ID] bash
cd /workspace/bin
```
3. DNNモデルの学習と保存を行うため、main_line.pyを実行する。実行の際に変数としてプロジェクト名とVersionを与える必要があります。
以下はtotinfo version2を対象にしてモデルを学習する実行。<br>
プロジェクトにもよりますが、モデルの学習にはある程度の時間がかかります。(30分~1時間くらい？)<br>
また、デフォルトでは全てのデータを訓練データとして使用するように設定してあり、訓練データサイズを変更する場合には以下のグローバル変数の編集を行います。<br>
    - TRAINING_FULL_DATA = False
    - TRAINING_SIZE_ = [任意のfloat値(0~1の範囲)]<br>

    また、モデルの保存にはSEED_VALUEという変数に格納された整数値をディレクトリ名に使用しており、同じ値のディレクトリが既に存在する場合にはモデルの学習は行われず処理が終了します。その場合、別の値を指定するか、既にあるディレクトリを削除してください。
```python3:main_line.pyの実行
python3 main_line.py totinfo 2
```
4. 学習したDNNモデルを用いて疑わしさスコアを算出するためにonlin_line.pyを実行します。onlin_line.pyでもSEED_VALUEという変数を用いてどのDNNモデルを用いるか指定します。適切な値に設定してください。また、同じSEED値でも複数のモデルが保存されている場合があるため、NNs_MODEL_NAME変数を用いてどのモデルを使用するか指定します。これも適切な名前に変更してください。本論文の実験では、基本的にDNNモデルの保存条件(precision, recall, TNR>=0.8)を満たした最初のエポックのモデルを使用しています。
```python3:onlin_line.pyの実行
python3 onlin_line.py totinfo 2
```
5. 結果の確認を行います。論文では一つのプログラムに対して10のモデルを作成したため、それぞれのモデルの結果の平均を最終的な実験結果としています。平均結果はWRITE_TOPN_PERCENTILEで指定するファイルに、それぞれのモデルの結果はOUTPUT_PER_SEEDで指定するファイルに書き込まれます。書き込まれる結果は故障を特定するために必要なコードの調査量であるTopN％です。
