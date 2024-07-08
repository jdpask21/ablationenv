# 概要
defects4jのカバレッジをjacocoを用いて収集するツールです.
カバレッジ収集の流れとしては下の感じになります:
1. defects4jのプロジェクトをコンパイルする.
2. クラスファイルにコード埋め込みを行う.
3. 各テストを行い, 各テストの実行ログを収集.
    - 実行するテストはgzoltarで集計する(all_tests.txtに保存).
    - 各テストと同じ名前のexecファイルが収集される.
4. 実行ログからカバレッジを抽出.

# 使い方
1. docker-compose でサービス構築
2. サービスにbashで入ってsetup.sh(/workspace/bin/setup.sh たぶんパス通ってる)実行
3. run_test.sh <project_id> <bug_id>
4. /workspace/clover/<project_id>/<bug_id>

# workspace/binの各ファイル
- checkout-compile.sh
    - 手順の1と2, gzoltarによるテスト集計を行う.
    - 引数にプロジェクト名とバグidを与える(e.g. "checkout-compile.sh Chart 2").
    - "/tmp/プロジェクト名/バグid"にコンパイル&埋め込み済みのプロジェクトが生成される.
    - コンパイル後の通常クラスファイルから埋め込み済みのクラスファイル(instrumentedの下)が生成され, これを実行すると実行ログが生成される.

- collect_coverage_parallelized.py
    - 手順の3, 4を行う.
    - 引数にプロジェクト名とバグidを与える(e.g. "collect_coverage_parallelized.py Chart 2").
    - all_tests.txtに保存されているgzoltarが集計したテストを並列実行する.
    - "workspace/clover/プロジェクト名/バグid"にカバレッジが生成される.
        - 失敗テストのカバレッジファイル名はfailから始まる.
        - "is_plane.txt"が生成されるが, ソースコードの内ステートメントがない行部分に1がついており, "is_plane.txt"で0の部分(ステートメントがある行部分)がカバレッジとして収集される. そのため, カバレッジファイルの行番号とソースコードの行番号が一致しない. が, test.pyの中ではそれを加味してカバレッジを読み込んでいるため行番号は一致するようになっている.
    - プログラムとしては残していますが、run_test.shを読んだらわかる通り、OpenCloverを使用する場合はこいつは使用しません。

- hoge.py
    - 実行回数カバレッジ情報を収集するプログラムです。run_test.shではデフォルトでこいつを使用
- hoge2.py
    - 通常のコードカバレッジ情報を収集するプログラムです。使う場合はrun_test.shでhoge.pyを実行した後、このスクリプト単体で実行してください。
- hoge3.py
    - 論文[Ablationとテストケース学習に基づく実行情報の欠落を利用した故障箇所特定](../../masterthesis2024.pdf), [Fault Localization with DNN-based Test Case Learning and Ablated Execution Traces](../../ISE2023-ikeda_t.pdf)で使用している実行トレースを収集するプログラムです。他と同様に使用する際はコメントアウトを解除してください。また、-cオプションを用いて実行することで、テスト実行の時間を省略します。（hoge.pyで収集したカバレッジレポートが必要です。）こちらも、-cオプションを用いて実行する場合は、run_time.shではなく、このスクリプト単体で実行してください。
        
- run_test.sh
    - checkout-compile.sh と collect_coverage_parallelized.py を実行するやつ.
    - 引数にプロジェクト名とバグidを与える
    ```shell:e.g.
    run_test.sh Lang 1
    ```
    - デフォルトではOpenCloverを使用するようにしています。基本的にOpenCloverでカバレッジ情報を収集してください。jacocoでは一部のプログラムでコードカバレッジ情報を収集することができないためです。
    - プロジェクトごとに依存情報が異なるため、いくつかのプログラム,シェルスクリプトでプロジェクトのVersion毎に変更する箇所があります。以下のファイルは変更必要箇所があるファイルです。必要箇所はプロジェクト名をエディターの検索機能を活用して特定してください。
        - run_test.sh
        - hoge.py (hoge2.pyはテスト実行機能を省いているので変更必要箇所はありません。ただし実行にはhoge.pyで収集したカバレッジレポートが必要です。)
        - hoge3.py
        - writemavenxml.py
        - xml_write_test.py

- SingleJUnitTestRunner.java
    - テスト関数単体で実行できるJUnitのやり方がわからなかったため, 無理やりテスト関数単体を実行するようなクラスを作った. collect_coverage_parallelized.pyではこのクラス経由で各テストを実行している.
    - antやJUnit自体にテスト関数単位で実行できる機能があるならこれはいらなくなる.

# mlのファイル
- test.py
    - 機械学習のやつ
    - tensorflowのバージョンは2.7.0でした
    - os.path.joinの部分は書き換えてください
- bugs_to_be_collected.txt
    - 対象となるバグ
- chunks
    - 各バグの修正された箇所(行番号とチャンクのセット)
    - "http://program-repair.org/defects4j-dissection/#!/"をスクレイピングした

# Clover使い方
1. 埋め込み
    ```
    java -cp /root/clover/lib/clover.jar com.atlassian.clover.CloverInstr --source <ソースファイルのjavaバージョン(mavenとかantの設定ファイルのどこかに載ってる)> -i clover.db -s 埋め込み前ソースディレクトリ -d 埋め込み後ソースディレクトリ
    ```
    上のコードで収集コードが埋め込まれたソースファイルを作れる. 最初からあるソースディレクトリsrcをtmp_srcにしておいてから -s tmp_src -d src でコンパイルすると楽. ソースディレクトリはプロジェクト依存だが, "defects4j export -p dir.src.tests" で取得できる.

2. コンパイル
    defects4j compileでコンパイル

3. テスト集計
    checkout-compile.shの最後と同様にgzoltarで集計

4. 実行
    ```
    java -cp /root/clover/lib/clover.jar:/workspace/bin:$(defects4j export -p cp.test) SingleJUnitTestRunner テスト項目(all_tests.txtの各行に該当) 
    ```
    クラスパスにclover.jar, SingleJunitTestRunner, プロジェクトのテストに必要な物たち を入れる

5. カバレッジレポート抽出
    ```
    java -cp /root/clover/lib/clover.jar com.atlassian.clover.reporters.xml.XMLReporter -if -l -i clover.db -o カバレッジレポートファイル
    ```

6. カバレッジ生成
   collect_coverage_parallelized.py の後半部分を用いてカバレッジレポートからカバレッジを生成する.


# 注意点
- 収集するプロジェクトによってディレクトリ構造がかなり変わる(同じプロジェクトでもバグidによって違う)ため, プロジェクトによってはcheckout-compile.sh と collect_coverage_parallelized.py のクラスファイルのパスなどを対応させる必要がある.
- cloverを使った方法は自動化されていないため(プロジェクト依存すぎて自動化できなかった), 手動でやる必要がある. 
- cloverはテスト毎にexecファイルを生成するタイプではなく, clover.dbに累積で実行ログが保存されるため, テスト実行のたびに"clover.dbからレポート抽出&clover.dbのリセット(埋込時に生成されるclover.dbをtmp_clover.dbとしてコピーしておき, 都度cp tmp_clover.db clover.db)"を行う必要がある. そのため並列でテスト実行ができず, めっちゃ時間がかかる.
