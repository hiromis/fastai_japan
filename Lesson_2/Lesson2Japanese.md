それでは皆さん、こんにちは、コーダのための実践的なディープラーニングにお帰りなさい。
これはレッスン2ですが、前回のレッスンでは最初のモデルのトレーニングを開始しました。
私たちは、トレーニングが実際にどのように行われるかは考えずに、より高いレベルで何が起こっているのかを見ていました。
そして、「機械学習とは何か」と「機械学習はどのように機能するのか」について学び、その働き方から、機械学習ができることには基本的な制限があることを認識し、一部について話しました。
また、機械学習モデルを訓練した後に、通常のプログラムのように振る舞うプログラムができあがることについても話しました：入力と中間にある何かと出力です。
今日はこの話を終わらせて、モデルを本番に投入する方法、そしてそれを行う際の問題点を見ていきます。
忘れないようにしておきたいのですが、2つの本，実際にはJupyter notebookですが，があります。
一つはfastbookレポ(オライリーの本のテキストがすべて含まれている実際のノート)で、私が話す内容をより詳細に確認できます。もう一つはコースv4レポジトリで、授業で使うものと同じですが理解を助けるために説明は省かれています。
ビデオを聞きながら、ビデオと本を行ったり来たりして、どちらか一方をやって、もう一方をやって、それを片付けて、コースv4のノートを見て、「よし、このセクションは何についてだったかな」と思い出して、コードを実行して、何が起こるかを見て、それを変更したりすることができるのです。
私たちはこのコードを見ていました。ここでは、どのようにして情報を渡してデータを作成したのか、おそらく最も重要なのはデータにラベルを付ける方法です。
そして、ラベル付けの重要性について話しました。この場合、この特定のデータセットでは、猫であれ犬であれ、最初の位置にある大文字か小文字かでわかります。
これがこのデータセット(readmeで教えてくれる)の仕組みです。
また、「valid percent equals 0.2」という考え方にも注目し、「これは何だっけ？バリデーションセットを作るものだ」といったような話もしました。そして私がさらに話したいことでもあります。 
まず最初に指摘しておきたいのは、この特定のラベリング関数は真か偽のどちらかを返すということです。
このデータセットには、後ほど見るように、37種類の猫と犬の実際の品種も含まれており、ファイル名から取得できます。
これら2つのケースでは、"猫なのか犬なのか？"、"ジャーマンシェパードなのかビーグルなのかラグドールなのか？"というカテゴリを予測しようとしています。カテゴリーを予測しようとしているとき、つまりラベルがカテゴリーであるとき、それを分類モデルと呼びます。
一方で、動物の年齢や身長などを予測しようとすることもあるかもしれません。
数字を予測しようとするときはいつでも、そのラベルは回帰と呼ばれる数字です。
いいですか？これがモデル分類と回帰の2つの主要なタイプです。
これは非常に重要な専門用語なのでしっかり覚えてください。


回帰モデルは、温度や場所などの1つ以上の数値量を予測しようとします。
これは少し紛らわしいですね。時々、人々は回帰という言葉を線形回帰と呼ばれる特定のモデルの略語のように使うことがあります。
しかし、線形回帰は回帰ではないので、大変紛らわしいです。
線形回帰は特定の種類の回帰に過ぎないことに注意してください。
回帰について話し始めると、多くの人は線形回帰のことを話していると思い込むでしょう。
さて、私はこのvalid_percent 0.2について話したいと思います。
先ほど説明したように、valid_percentはデータの20パーセントを別のバケツに入れておき、モデルを訓練するときにはそのデータを使いません。
そのデータは、モデルがどれだけ正確であるかを示すためだけに使用されます。
ですから、もしあなたがあまりにも長い間、あるいは十分なデータを持たずに、あるいはパラメータが多すぎるモデルを使ってトレーニングした場合、しばらくするとモデルの精度は実際に悪くなります。
これをオーバーフィットと呼びます。そこで、オーバーフィットしていないことを確認するために検証セットを使用します。
次に見たコードの行はこの行で、Learnerと呼ばれるものを作成しました。
Learnerとは、基本的にはデータとアーキテクチャ、最適化している数学的な関数を内包します。
これについてはもう少し詳しく説明しますが、基本的にはこの特定の関数ResNet34は、コンピュータビジョンの問題に非常に適した特定のアーキテクチャの名前です。
実際の名前は ResNet で、34 はレイヤーの数を表しています。
つまり、ここではより大きな数値のものを使うことで、より多くのパラメータを得る，つまりより複雑なモデルを作成することもできますが、訓練に時間がかかり、より多くのメモリを必要とし、オーバーフィットしやすくなります。
今はこの部分に注目したいと思いますが、ここではmetrics=error_rateです。
ここでは、あなたが使いたい関数を列挙しています... 
データと一緒に呼び出されたい関数をリストアップします。 
検証データと一緒に、各エポックの後にプリントアウトします。エポックとは、データセット内のすべての画像を一度だけ見ることを指します。 
データセット内のすべての画像を一度見た後に、モデルの情報を表示します。最も重要なのは、これらのメトリクスを呼び出した結果をプリントアウトすることです。 


つまり、我々のメトリックは検証セットを使用して予測の品質を測定する関数であり、誤差率は1から精度を差し引いたものです。
アーサー・サミュエルはこの機械学習における重要な考えを持っていました。モデルの性能がどれくらい良いかを把握するためには何らかの方法が必要で、パラメータを変更したときに、どのパラメータのセットが性能を良くするか悪くするかを把握することができます。
損失は必ずしもメトリクスと同じとは限りません。
その理由は少し微妙で、今後のレッスンで数学を掘り下げていくと詳細が見えてきますが、基本的には関数が必要で、パラメータを少しだけ上げたり下げたりして、損失が少し良くなったか悪くなったかを見ることができる損失関数が必要です。予測が犬から猫に変わるほどパラメータを変化させるわけではないので精度やエラーレートでは不十分なのです。
予測が同じならエラーレートも同じはずです。
損失とメトリックは密接に関連していますが、メトリックはあなたが気にしているもので、損失はパフォーマンスの測定値としてコンピュータがパラメータをどのように更新するかを決定するために使用しているものです。
なので、検証セット上のメトリックを見てオーバーフィッティングをチェックしています。
FastAIは常に検証セットを使用してメトリクスを出力します。オーバーフィッティングは重要で、学習しているデータだけでなく、学習アルゴリズムが見たことのないデータに適合するモデルをどうやって見つけるかということです。
つまり、モデルが基本的に "不正行為 "をしている場合、オーバーフィットと言えます。
モデルは、「この正確な写真を見たことがある、それは猫の写真だと覚えている」と言うことでごまかすことができます。
つまり、一般的に猫がどのように見えるかを学習していないかもしれませんが、画像1〜4と8が猫で、画像2〜3と5が犬であることを覚えていて、実際には猫がどのように見えるかについては何も学習していません。
このような不正行為は避けようとしているのですが、特定のデータセットを記憶させたくないのです。
そこで、検証データを分割して、画面に表示されている単語のほとんどは、本の中の単語ですので、コピーして貼り付けました。
検証データを分割して、モデルが学習中にそれを見ないようにすれば、完全にデータに汚染されないので、ごまかすことはできません。
そうではありません。ごまかすことはできます。ごまかす方法としては、モデルをフィットさせて結果と検証セットを見て、何かを少し変更して別のモデルをフィットさせて検証セットを見て、何かを少し変更して、検証セットが一番良さそうなものが見つかるまで100回くらい繰り返してみることができます。
しかし、今ではバリデーションセットに適合しているかもしれませんね。
ですから、もしこれを本当に厳密に行いたいのであれば、テストセットと呼ばれる3番目のデータを用意しておくべきです。
これは実際には、プロジェクト全体が終了するまで見ることはありません。
これは、Kaggleのような競争プラットフォームで使われているものです。
Kaggleでは、競技が終了した後、見たことのないデータセットに対してパフォーマンスが測定されます。


これは本当に有益なアプローチで、自分でモデリングをしていなくても、実際にそうするのは素晴らしいアイデアです。
もしあなたがベンダーを見ていて、今日はIBMかGoogleかMicrosoftにしようと決めようとしていて、彼らのモデルがどれだけ素晴らしいかを見せてくれているなら、あなたがすべきことは、「オーケー、あなたはモデルを構築してきて、私は私のデータの10％を保持して、あなたには全く見せないようにします。
さて、検証セットとテストセットを引き出すのはちょっと微妙ですが。
これは簡単な小さなデータセットの例で、これはレイチェルが書いた、効果的な検証セットの作成についての素晴らしいブログ記事から来ています。
基本的には、ある種の季節のデータセットを持っていることがわかります。
さて、"Ok, fas.ai, I want to make a my dataloader using valid_percent of 0.2 "と言うと、次のようになります。
ランダムにいくつかのドットを削除してくれますよね？
これはあまり参考になりませんが、ドットが他のドットの真ん中にあるので、ズルをすることができます。
実際に何が起こるかというと、これは日付ごとの売上高ですが、来週の売上高を予測したいと思います。
14日前、18日前、29日前の売上ではありません。
ここで効果的な検証セットを作成するために実際に必要なのは、ランダムに行うのではなく、最後の部分を切り落とすことです。
これは全てのKaggleコンテストで起こることで、例えば、時間が関係していて、予測しなければならないのは、最後に与えられたデータポイントの２週間後くらいのことで、これはテストセットでも行うべきことなんだ。
そうすれば、再学習も何もできなくなるでしょう、なぜならそれが実際に起こることだからです。そうなんですか？
質問があるんですが、オーバーフィットは学習誤差が検証誤差を下回ると説明する人がいるのを聞いたことがありますが、この経験則はあなたの経験則と大体同じですか？
そうですか、それは素晴らしい質問ですね。
つまり、彼らが言いたいのは、トレーニング・ロスとバリデーション・ロスのことだと思います。
我々は訓練誤差を表示しないので、各エポックの終わりに訓練セットの損失関数の値と検証セットの損失関数の値を表示します。
そして、もしあなたが十分に長い間訓練していれば、それはそうですので、もしそれがほとんど訓練であれば、訓練の損失は減少し、検証の損失は減少します。
なぜなら、定義上、損失関数は、損失関数が低いほど良いモデルであるように定義されているからです。
オーバーフィットを始めると、トレーニングの損失は下がり続けますよね？


なぜそうしないの？
パラメータはどんどん良くなっていますが、実際には学習セットの特定のデータポイントにフィッティングを始めたため、検証の損失は増加し始めます。
しかし、実際にはトレーニングセットの特定のデータポイントにフィッティングを開始したため、検証の損失は増加し始めます。
バリデーションセットが良くなるわけではなく、悪化し始めるでしょう。
しかし、これは必ずしもオーバーフィッティングをしているとは限りませんし、少なくとも悪い意味でオーバーフィッティングをしているわけではありません。
損失関数についてもっと学ぶ必要があるので、このようなことがどのように起こるのかを数学的に説明するつもりはありませんが、これから説明していきます。
しかし、今のところは、重要なのはメトリックが悪化していることであって、損失関数が悪化していることではないということを理解しておいてください。
素晴らしい質問をありがとうございます。
次に学ぶべき重要なことは、伝達学習と呼ばれるものです。
つまり、次の行ではlearn.fine_tuneと書いてあるんですね。
なぜlearn.fine_tuneと書いてあるのでしょうか？
fine tuningとは、転移学習をしているときに行うことで、転移学習とは、元々訓練されたものとは異なるタスクに対して、事前に訓練されたモデルを使用することです。
ということで、専門用語を理解するためにもっと専門用語を。
それを見てみましょう。
事前訓練モデルとは何ですか？
どうなるかというと... 我々が使っているアーキテクチャは ResNet-34と呼ばれていますよね？
これは数学的な関数であり 多くのパラメータを持っています 機械学習を使って適合させます 
イメージネットというビッグデータセットがありますが、そこには130万枚の写真が入っていて、キノコでも動物でも飛行機でもハンマーでもなんでもいいんです。
このImageNetで誰が最高の精度を出せるかを競う大会が毎年開催されていましたが、かつてはその大会も開催されていました。
その中で、本当に良い結果を出したモデルは、パラメータの特定の値を取って、インターネット上で誰でもダウンロードできるようにしていました。


これをダウンロードすると、ただのアーキテクチャではなく、訓練されたモデルを手に入れることができます。
画像の中にある1000のカテゴリを認識できるモデルを持っています。
これは、たまたまその1000クラスを正確に認識するものが必要な場合を除いては、あまり有用ではないでしょう。
しかし、実際には、その重みから始めて、データ上でさらにエポックを訓練すれば、事前に訓練されたモデルから始めなかった場合よりもはるかに正確なモデルができあがります。
この移動学習のアイデアは、直感的に理解できますよね？
ImageNetにはすでに猫と犬が登録されていて、これは猫でこれは犬だと言うことができますが、ImageNetには登録されていない多くの犬種を認識するようなことをしたいと思います。
猫と犬、飛行機とハンマーを認識するためには、金属はどんな形をしているのか、毛皮はどんな形をしているのか、などを理解する必要があります。
毛皮はどんな形をしているのか？
耳はどんな形をしているか？
そうすることで、「この動物のこの品種、この犬のこの品種は耳がとがっている」「これは金属だから犬ではない」というようなことが言えるようになります。
これらの概念はすべて、事前に訓練されたモデルによって暗黙的に学習されます。
ですから、事前に訓練されたモデルから始めれば、これらすべての特徴をゼロから学習する必要はありません。
これがfastaiライブラリの重要な焦点であり、このコースの重要な焦点です。
質問があります。損失、エラー、メトリックの違いについて少し混乱しています。
確かに、エラーはメトリクスの一つの種類に過ぎないので、様々なラベルが考えられます。
例えば、猫や犬の年齢を予測できるモデルを作成しようとしていたとしましょう。
そこで、あなたが使うかもしれないメトリックは次のようなものです：平均して、何年ずれていたか？
これが指標になります。
一方、もしこれが猫か犬かを予測しようとしている場合、あなたのメトリックは次のようになります：私は何％の確率で間違っているか？
後者のメトリックは誤差率と呼ばれています。


エラーは特定の指標の一つです 
どれだけうまくいっているかを測るもので、あなたが最も気にしていることです。
つまり、関数を書いたり、fastaiの定義済みの関数を使ったりして、どれだけうまくいっているかを測定します。
損失はレッスン１で話したことなので、簡単に要約しますが、覚えていない場合はレッスン１に戻ってください。
アーサー・サミュエルは、機械学習モデルにはパフォーマンスの指標が必要で、それを見ることができると話しています。
そして、先ほども述べたように、パラメータを少しだけ上下させても全く変化しない指標もあります。
そのため、パラメータを調整してより良いパフォーマンスの指標を見つけるという目的には使えません。
損失関数と呼ばれる別の関数を使用することがよくあります。損失関数とは、アルゴリズムがパラメータをより良いものにするために使用するパフォーマンスの指標で、これはあなたが気にしているメトリックに近いものですが、パラメータを少し変えると損失は常に少し変化します。
このように、手を振っている場合が多いのですが、これがどのように機能するのかを計算する必要があるため、次のレッスンではそれを学習します。
素晴らしい質問をありがとうございました。
fine tuningとは、スライドではなく写真を表示したままにしておくことです。
fine tuningとは、事前学習したモデルのパラメータを、事前学習で使用したものとは別のタスクを使用して追加エポックの学習を行うことで、重み（これはパラメータと言うべきかどうかは微妙なところですが）が更新される伝達学習の手法です。
つまり、事前学習のタスクはImageNetの分類であり、別のタスクは猫と犬を認識することになるかもしれません。
デフォルトでfastaiがfine tuningを行う方法は、1つのエポックを使用することです。
1つのエポックでモデルの特定の部分をフィットさせ、データセットの特定の部分が機能するようにします。
そして、モデル全体を適合させるために、あなたが要求した数だけエポックを使用します。
これは、もう少し上級者の方のために、レッスンの後半でどのように機能するかを正確に見ていきます。
では、なぜ転移学習はうまくいくのでしょうか？
彼らは実際に2012年のImageNetの受賞者であり、興味深いことに、彼らの主要な洞察はモデルの内部で何が起こっているかを可視化する能力に由来しています。
このように、可視化は素晴らしい結果を得るために非常に重要であることがよくわかります。

彼らができたのは、34層のResNet34と言ったのを覚えていますか？
前回のコンクールで優勝したAlexNetと呼ばれるものを調べましたが、これは7層しかありませんでした。
当時はそれは巨大なものと考えられていましたので、7層モデルを使って、最初の層のパラメータはどのようなものかと考えました。
そして、その絵をどうやって描くかを考えました。
最初の層にはたくさんの特徴がありましたが、ここには9つの特徴があります。
これがそのうちの９枚の絵です 
そのうちの一つは、左上から右下までの対角線を認識できるものでした。
そのうちの一つは左下から右上への対角線を見つけることができました。
オレンジの上から青の下へのグラデーションを 見つけることができるものもありました 
彼らの中には、緑色のものを見つけることができる人もいました。
これらの9つはそれぞれフィルターや機能と呼ばれています。
これらのフィルターや特徴をそれぞれ見てみました。これらが実際に何を意味するのかについては次の授業で数学的に学びますが、今のところはそれらを認識して、対角線を見るものとグラデーションを見るものがあり、そのフィルターにマッチする写真の具体的な例をイマゲネットの実際の画像から見つけました。
左上のフィルターにはそのフィルターにマッチする実際の写真の９つのパッチがあります。
緑のフィルターでは緑のフィルターにマッチした写真のパーツを使用しています。
ここで注目すべきは、グラデーションや色や線のパッチを認識できるものは、imagenetだけでなく他の多くのタスクにも使える可能性があるということです。
このようなことができるものは、他の多くのコンピュータビジョンのタスクにも使える可能性があります。
これはレイヤー2で、レイヤー2はレイヤー1の機能を利用してそれらを組み合わせています。
これはエッジだけではなく、コーナーや繰り返しの曲線パターン、半円や全円を見つけることができます。
例えば、ここではレイヤー1の後のレイヤーを正確に可視化するのは難しいと思います。
フィルターがどのように見えるかの例を示す必要があります。


このレイヤー2の円形フィルターが活性化した写真の一部の例をご覧ください。
ご覧のように円形のものが見つかっています。
興味深いことにこのぼやけたようなグラデーションは夕日を見つけるのに非常に適しているようです。
そして、この繰り返しの垂直パターンは、カーテンや小麦畑などを見つけるのに非常に適しています。
このようにして、レイヤー3をさらに進めると、レイヤー2にあるすべての種類の特徴を組み合わせることができるようになります。
ここでは12個の特徴しか見ていませんが、実際には数百個の特徴があるでしょう。
alex netでは正確には覚えていませんが、たくさんあります。
しかし、第2層の機能を組み合わせて第3層に到達する頃には、すでにテキストを見つける機能を持っています。
これはテキストを含む画像の断片を見つけることができる機能です。
幾何学的なパターンを繰り返し見つけることができるものをすでに持っています。
これは単なる特定のピクセルパターンのマッチングではありません。
これは意味的な概念のようなものです。
繰り返しの円や繰り返しの四角や繰り返しの六角形を 見つけることができます 
すごいですね。
つまり、これは本当に計算機のようなもので、単にテンプレートをマッチングするだけではありません。
ニューラルネットワークはあらゆる計算可能な関数を解くことができることを覚えておいてください。
だから、確かにそれができます。
だから、第4層は第3層のフィルタを一度にすべて組み合わせることができます。
そして第4層では、例えば犬の顔を見つけることができるものができます。
このように、各レイヤーでは、より応用的に洗練された機能が得られるようになっています。


だからこそ、これらのディープ・ニューラル・ネットワークは信じられないほど強力なものになるのです。
それはまた、転移学習が非常にうまく機能する理由でもあります。
本を見つけることができる何かが必要だからです。
ImageNetには本のカテゴリがないと思います。
実際にはすでにテキストを見つけることができるものがありますが、それは以前のフィルターとして、図書館などのカテゴリや本棚を見つけるために使用しているのではないかと思います。
転移学習を使うと、事前に学習した機能をすべて活用して、これらの機能や既存の機能を組み合わせたものを見つけることができます。
これが、従来のアプローチに比べて、転移学習がより速く、より少ないデータでできる理由です。
ここで重要なことは、コンピュータビジョンのためのこれらの技術は写真を認識するのが得意なだけではないということです。例えば、これらは時間の経過とともに周波数を表現して画像化した音です。
9:45ですが、すぐに休憩しますか？
私の記憶が正しければ、彼は基本的にマウスを動かしたときに、マウスを動かした場所の絵を描いていました。
Splunkで彼が実際にやったことは、コースのプロジェクトとして、レッスン1で見たのと全く同じアプローチでこの絵を使って不正防止モデルを作成できるかどうかを試してみたことです。

もう一つの素晴らしい例は、様々なウイルスを調べて画像に変換したものです。これは論文からの引用ですが、本を見てください。
この本の中には、これまでに見てきた全ての用語、最も重要な用語のリストが載っています。
アーサー・サミュエルズの全体的なアプローチでは、アーサー・サミュエルズの用語を我々の用語に置き換えて、入力としてパラメータを含むアーキテクチャ、パラメータとデータを含む入力としてのデータ、アーキテクチャとパラメータがモデルとなり、予測値を計算するために使用される入力で、損失関数を用いてラベルと比較されます。
これで本の第1章は終わりです。
アンケートを見ることは本当に重要です。アンケートは、あなたがこの本やこの章から、私たちが望んでいることを持ち帰ったかどうかを確認するためのものです。
この本を読んで、わからないことがあれば、その答えはテキストの中にあるので、本の最初の方に戻って、その章の中に答えが書いてあります。
最初の2～3章では、質問に答えるために、章を見るだけでは不十分で、実際に自分で考えたり、実験したり、グーグルで調べたりしなければならないような、とてもシンプルな内容になっています。
後の章では、数日から数週間かかるかもしれない重要なプロジェクトを紹介していますので、それらをチェックしてみてください。
もしあなたが本当にこの本を最大限に活用したいのであれば、各章の後に時間をかけて自分のプロジェクトや私たちが提供する本の中で実験をして、新しいデータセットでノートをやり直せるかどうかを確認してください。


第一章は少し難しいかもしれませんが、次の第二章では、絶対にできるようになるでしょう。
それでは５分間の休憩を取り、サンフランシスコ時間の９時５５分に戻ります。
それでは、みなさん、ようこそ、戻ってきてください。
確かに、フィルターは独立しているのでしょうか？つまり、フィルターが事前に訓練されていると、fine tuningすると、以前の画像の特徴を検出しにくくなるのではないでしょうか？
これは素晴らしい質問ですね。私が正しく理解していると仮定すると、イマジェネット・モデルから始めて、犬と猫で数エポックの間fine tuningをして、犬と猫を認識するのに非常に優れたものを得たとしても、その後はイマジェネット・モデルとしての性能はかなり低下します。
これは、文献では破局的忘却と呼ばれていますが、以前に見たものとは異なるものについての画像を多く見ると、以前に見たものが何であるかを忘れてしまうという考え方です。
ですから、新しいタスクが得意で前のタスクが得意なものをfine tuningしたい場合は、前のタスクの例を入れ続ける必要があります。
パラメータとハイパーパラメータの違いは何ですか？
犬の画像を入力として与え、モデル内のバッチサイズのハイパーパラメータを変更している場合、パラメータの例は何でしょうか？
つまり、パラメータとは、アーサー・サミュエル氏がレッスン1で説明したもので、モデルの動作やアーキテクチャの動作を変更するものです。
ニューラルネットワークと呼ばれるものは、何でもできる無限の柔軟性を持った関数で、パラメータを変更することで、あることと別のことを実行させることができます。
パラメータとは、関数に渡す数値のことで、関数に渡す数値には2種類のタイプがあります。
ニューラルネットではありませんが、アーサー・サミュエルが60年代前半から50年代後半に使っていたようなチェッカープログラムのようなものの例では、これらのパラメータは次のようなものだったかもしれません：駒を取る機会と、盤面の端に到達する機会がある場合、どちらか一方ともう一方のどちらかをどれくらいの価値があると考えるべきか、ということです。
2倍重要なのか3倍重要なのか、2対3はパラメータの例です。
ニューラルネットワークでは、パラメータはもっと抽象的な概念なので、パラメータが何であるかの詳細な理解は次のレッスンで説明しますが、基本的な考え方は同じです：パラメータは数値であり、悪性腫瘍を認識するように、猫と犬を認識するように、白黒の写真をカラー化するように、モデルの動作を変化させるものです。
一方、ハイパーパラメータは、関数にどのような数値を渡すかの選択であり、実際のフィッティング関数に渡して、フィッティング処理がどのように行われるかを決定します。
このコースのペーシングが気になる」という質問があります。
すべての材料がカバーされていないのではないかと心配です。" 全てを網羅するとはどういうことかにもよるが 
世界のすべてをカバーすることはできませんから、できる限りのことはカバーします。
7つのレッスンでできることをカバーします。


本の全体を２コースか３コースのどちらかでカバーする予定です。
これまでは、この本の内容をカバーするには2つのコースが一般的でしたが、500ページにも及ぶかなり大きな本なので、どうなるか見てみましょう。
2つのコースというと、14レッスンということですか？14レッスン、そうですね、本を全部読むには14レッスンか21レッスンになります。
でも、最初のレッスンが終わる頃には、自分で本を読むことがより役立つことを理解し、フォーラムで一緒に質問をしたりできるコミュニティを得られるようになっているはずです。
そのためには、ディープラーニングの能力と限界とは何かを理解する必要があります。本番に投入することに意味があるプロジェクトとはどのようなものなのでしょうか？
この本とこのコースで言及すべき重要なことの1つは、最初の2～3レッスンと章で、コーダーのためだけではなく、すべての人のために設計された多くの内容があるということです。
ディープラーニングを機能させるために知っておくべき実践的なことについての情報がたくさんあります。
そのうちの1つは、「ディープラーニングは今のところ何に向いているのか」ということです。
本に書いてあることを要約しますが、Fastaiには4つの主要な分野があります。コンピュータビジョン、テキスト、表形式、そして私がここで "Recsys "と呼んでいるもの、レコメンデーションシステムのためのものです。 
申し訳ありませんが、もう一つ質問があるのですが、ImageNetのもの以外に、我々が使用できる事前に訓練された重みはありますか？ 
もしあるとしたら、いつ他のものを使うべきで、いつイマゲネットを使うべきなのでしょうか？
ああ、それは本当に素晴らしい質問ですね。
そうですね、事前に訓練されたモデルはたくさんありますし、それらを見つける一つの方法としては...。
そして、あなたは今、私たちに見せてくれています... 
そうですか... 
彼らを見つける素晴らしい方法の一つは、モデル動物園を調べることができます それは多くの異なるモデルを持っている場所の一般的な名前です 
ここにたくさんのモデル動物園があります。
あるいは、訓練済みのモデルを探すこともできます。
残念ながら、私が望むほど多様性はありませんが、ほとんどはまだImageNetや類似の一般的な写真に掲載されています。
例えば、医療用画像はほとんどありません。 


人々がドメイン固有の事前学習モデルを作成する機会はたくさんありますが、伝達学習に取り組んでいる人が十分ではないため、まだ十分に行われていない分野です。
ディープラーニングはスプレッドシートやデータベースのテーブルのような表形式のデータに非常に適しています。
ディープラーニングは特にそういったものに非常に適しています。
テキストについては、分類や翻訳のようなものに非常に優れています。
会話には向いていないので、これが多くの企業にとって大きな失望となっていました。会話ボットのようなものを作ろうとしましたが、ディープラーニングは正確な情報を提供するのが得意ではありません。
レコメンデーションシステムの共同フィルタリングの大きな問題の一つは、ディープラーニングが予測を行うことに焦点を当てていることですが、それは必ずしも実際に有用なレコメンデーションを作成することを意味するわけではありません。
それが何を意味するのかは、しばらくしてから見ていきましょう。
ディープラーニングはマルチモーダルも得意としており、複数の異なるタイプのデータを持っている場合には、テキストカラムや画像を含む表形式のデータを持っているかもしれないし、協調フィルタリングのデータを持っているかもしれないし、それらをすべて組み合わせることもディープラーニングが得意とすることです。


例えば、写真にキャプションを付けることはディープラーニングが得意とすることですが、正確さはあまり得意ではありません。
これは2羽の鳥の写真ですが、実際には3羽の鳥の写真であることがわかります。
タンパク質の解析をしているのも素晴らしいですね。
異なるタンパク質を異なる言葉として考え、それらがある種の状態や意味を持っている配列にあると考えれば、ULMFitはタンパク質の解析に非常に適していることがわかります。
このように、多くの場合、創造性を発揮することが重要です。
あなたが構築しようとしている製品のように、ディープラーニングがうまく機能するかどうかを決めるには、最終的には試してみるしかありませんが、検索すれば、似たようなことを試した人の例が見つかると思います。
例えば、レコメンデーションと予測は必ずしも同じものではないという協調的なフィルタリングの問題についてお話しました。
これは、例えばAmazonでよく見られます。
私はTerry Pratchettの本を買ったのですが、Amazonは数ヶ月間、私にもっとTerry Pratchettの本を買わせようとしました。
これは、彼らの予測モデルが、ある特定のテリー・プラットシェットの本を買った人は、他のテリー・プラットシェットの本も買う可能性が高いと言っていたからに違いありません。
しかし、これは私の購買行動を変えようとしているのでしょうか：おそらくそうではないでしょう、右、もし私がその本が好きならば、私はすでにその著者が好きだと知っているし、彼らはおそらく他のものを書いていると知っているので、私はとにかくそれを買いに行くでしょう。
だからこれは、Amazonはおそらく非常にスマートではないような例になるでしょう、ここで彼らは実際に推薦を最適化する方法を実際に考え出すのではなく、実際に私に共同フィルタリングの予測を示しています。


ですから、最適化されたレコメンドとは、あなたの地元の人間の本屋さんがするようなものでしょう。
だから、おすすめと予測の違いは超重要なのです。
そこで私は、モデルの解釈に関する本当に重要な問題について話したいと思い、そのためのケーススタディのために、この論文のモデルである、今、実際に超重要なものを選んでみようと思いました。
このコースでは、論文の読み方を学ぶことにしています。
これは、高温多湿がCOVID-19の感染を減少させるという論文です。
もしこの論文の主張が本当ならば、これは季節性の病気であることを意味し、もしこれが季節性の病気であれば、大規模な政策的な意味合いを持つことになります。
これがどのようにモデル化されたのかを調べて、このモデルをどのように解釈するかを理解しましょう 
これが論文からの重要な図です 中国の100の都市の気温を 摂氏でプロットしました もう一つの軸にはRをプロットしました 
各人がこの病気に感染した場合 平均何人が感染するかを表しています 
つまり、Rが1以下であれば、この病気は広まりません。
もしＲが２より高ければ 信じられないほど早く広がります 
基本的に、Rが高くなると、指数関数的に感染が拡大します。
このケースでは、ここに最適な線をプロットしています 
そして、彼らは、Rが1.99から0.023倍の温度を引いたものであるという公式に基づいて、ある特定の関係があると主張しています 
この図を見ていて私が懸念しているのは、これは単なるランダムなものかもしれない、もしかしたら全く関係がないのかもしれない、ということです。
それを見る簡単な方法としては、実際にスプレッドシートを使ってみるという方法があります。
これがスプレッドシートです。
私がやったことは、このデータを見て、平均気温が何度かを推測してみました。
約5度だと思います 
摂氏の標準偏差は？


たぶん同じように5くらいだと思います。
そして、Rについても同じことをしました。
平均Rは私には1.9くらいに見えると思います。
そして、Rの標準偏差はおそらく0.5くらいだと思います。
そこで、私はここにジャンプして、ランダムな正規値、つまり正規分布からのランダムな値を作成しました。
これは100の都市のデータセットの中にある都市の例です 
摂氏9度、Rが1.1の都市は、摂氏9度、Rが1.1で、このあたりになります。
だから、この式を100回コピーしました。
ここに中国の100の都市がありますが、これは気温とRの間に関係がないと仮定しています。
これらはただの乱数なので、毎回、再計算しています コントロール・イコールを押すと、再計算されます 
ランダムなので、違う数字が出てきます。
上の方には、全温度の平均と全Rの平均、全温度の平均が表示されています。
そこで私がしたことはこれらの乱数をここにコピーしたことです 
実際にやってみましょう 
100個の乱数をコピーして ここに貼り付けます ここに貼り付けます 
これで１ 2 3 4 5 6 100個の都市のグループが６つできました 
これでランダムに変化するのを止めましょう石に固定しておきます
さて、これで貼り付けました。気温とRの関係が全くないとしたら、100の都市はどのようになるのか、6つの例が出てきました。
6つの例の平均気温とRはそれぞれの都市のものです 
私がした事は、ここで見て下さい、少なくとも最初の例はプロットしました、いいですか？ 


この場合、実はわずかに正の傾きがあることがわかります。
私は実際に、Microsoft Excelの傾き関数を使って、それぞれの傾きを計算しました。
あなたは、実際に、この特定のケースでは、 ただランダムであることがわかります - 5回、負の値を示しています そして、それは彼らの0.023よりもさらに負の値を示しています 
このように、私たちの直感と一致しています。つまり、この線の傾きは、完全に偶然に起こることが多いということです。
実際の関係を示しているようには見えません 
もし、この傾きをもっと自信を持って 示したいのであれば、もっと多くの都市を 調べる必要があるでしょう 
ここに3,000個のランダムに生成された数字があります。
傾きが0.00002であるのがわかりますね？
これはほぼ正確には0ですが、これは我々が期待するもので、実際にはCとRの間には関係がなく、この場合は関係がありません。
では、ランダムに生成されたたくさんの都市を見てみると、ああ、そうだ、傾きはない、と言えるでしょう。
しかし、今回のように１００個の都市だけを見た場合には、非常に多くの場合、全く偶然に関係性があることがわかります。
だから、それを測定する必要があります。
これを測定する一つの方法は、p値と呼ばれるものを使うことです。
p値、ここではp値がどのように機能するかを説明します：まず帰無仮説と呼ばれるものから始めます。
帰無仮説とは、基本的には出発点の仮定のことです。
我々の出発点の仮定は、温度とRの間には何の関係もないというものかもしれません。
そして、いくつかのデータを集めて、（レイチェル：Rとは何か説明してくれましたか？ ジェレミー：はい、しました)
Rはウイルスの透過性です。
そして、独立変数と従属変数のデータを収集します - この場合、独立変数とは、従属変数を引き起こすと思われるものです。
ここでは、独立変数は温度で、従属変数はRです。


ここでデータを収集しました - この例で収集したデータがあります。そして、偶然にも0.023の傾きであるこの関係性を、何％の割合で見ることができるでしょうか？
これまで見てきたように、これを行う1つの方法は、いわゆるシミュレーションと呼ばれるもので、乱数を生成して、100組の乱数のペアを何回か繰り返して、この関係がどのくらいの頻度で見られるかを見るというものです。
しかし、実際にそれをする必要はありません。
実際には、この数値を求めるのに使える簡単な方程式があります。
これは基本的にはこのようになります 
最も可能性の高い観測がありますが、この場合は、気温と気温の間に関係がない場合です。
この場合、最も可能性の高い傾きはゼロになりますが、偶然にも正の傾きが得られることもあれば、かなり小さな傾きが得られることもあれば、大きな負の傾きが得られることもあります。
このように、数字が大きくなるほど、それが正の側であろうと負の側であろうと、その可能性は低くなります。
私たちの場合、どのくらいの頻度で0.023以下の負の値を得ることができるのか、というのが私たちの質問でした。
それは実際にはこの下のどこかになるでしょう。
これをウィキペディアからコピーしました。正の数値を探しているところです。
これがp値です。これは数学のことは気にしませんが、データからこの数値、つまりp値を直接計算するのに使える簡単な小さな方程式があります。
これは、ほとんどすべての医学研究の結果がこのような形で示される傾向があり、人々はこのp値の考え方に注目しています。
実際に、この特定の研究では、後ほどご紹介するように、p値が報告されています。
おそらく、多くの人が前世でp値を見たことがあるでしょう。
それらは様々な分野で出てきます。
ここからが問題なのですが、これはひどいものです。
使うべきではありません。
私だけを信用しないでください。
アメリカ統計協会を信じろ 


彼らはp値について6つのことを指摘していますが、その中には次のようなものがあります：p値は仮説が真である確率を測るものではない、あるいはデータがランダムな選択だけで作られた確率を測るものではない、ということです。
これは、より多くのデータを使用した場合、100の都市ではなく3000の都市を無作為にサンプリングした場合、得られる値ははるかに小さくなることがわかったからです。
つまり、p値は関係性の大きさを教えてくれるだけではなく、実際にはそれらを組み合わせて、どれだけのデータを収集したかを教えてくれるのです。
つまり、仮説が真である確率を測定するものではないのです。
したがって、結論や政策決定は、P値がある閾値を通過したかどうかに基づいて行われるべきではありません。
P値は結果の重要性を測定するものではありません。なぜなら、繰り返しになりますが、P値は、あなたが多くのデータを収集したことを教えてくれるだけで、結果が実際に実用的なものであることを教えてくれるわけではないからです。
それだけでは、エビデンスの良い尺度にはなりません。
フランク・ハレルという人がいますが、彼の本を読みました。
彼は生物統計学の教授で、このことについて多くの素晴らしい論文を持っています。
彼は帰無仮説検定とp値が科学に大きな害を与えていると言っています。
彼は「帰無仮説有意性検定は決して機能しなかった」という別の記事を書いています。
私はp値が何であるかを示したのは、なぜ機能しないのかを知るためであって、それを使えるようにするためではありません。
しかし、それらはいつも出てくるので、機械学習の超重要な部分です。
人々が「あなたの薬が効いたかどうか、疫学的な関係があるかどうか、何でもいいから、こうやって決めよう」と言っているときに、これが機械学習の超重要な部分なのです。
そして確かに、この論文にはp値が出てきます。
この論文では、重回帰の結果が示されています。
彼らは、p値が0.01以下の関係の隣に3つの星をつけています。
0.01以下のような小さなp値については、何か役に立つことがあります。
これは、私たちが見ているものは、おそらく偶然には起こらなかったのではないでしょうか？
人々がいつも犯している最大の統計的ミスは、p値が0.05以下ではないことを見て、関係が存在しないという誤った結論を下してしまうことですよね？


これでは意味がありません。なぜなら、3つのデータポイントしかなかったとすると、どの仮説に対してもp値が0.05未満になるような十分なデータはほぼ確実に得られないからです。
そこで、確認する方法としては、遡って、もし私が全く逆の帰無仮説を選んだとしたらどうだろうか？
私の帰無仮説が、温度とRの間に関係があるとしたらどうでしょうか？
その帰無仮説を棄却するのに十分なデータがあるでしょうか？答えがノーなら、結論を出すのに十分なデータがないだけですよね？
この場合、彼らは温度とRの間に関係があると確信するのに十分なデータを持っています。
このグラフを見て、エクセルでちょっと裏ワザをしてみたんですが、これは、ランダムなものかもしれないと思ったんです。
問題はここにあります 
このグラフは、一変量関係と呼ばれるものを示しています。
一変量関係は、1つの独立変数と1つの従属変数の間の関係を示します。
しかし、このケースでは、多変量モデルを用いて、気温、湿度、一人当たりGDP、人口密度を調べました。
なぜそうなるのでしょうか？
なぜこのようなことが起こるのかというと、青い点の変動はランダムではないからです。
異なっているのには理由がありますよね？
その理由としては、例えば、密度の高い都市ほど透過率が高く、湿度の高い都市ほど透過率は低くなります。
多変量モデルを使うと結果に自信が持てるんですよね？
しかし、アメリカ統計協会が指摘しているp値は、これが実際に重要かどうかを教えてくれるものではありません。
これが実用的に重要かどうかを教えてくれるのは、発見された実際の傾きです。
この場合、彼らが出してきた方程式は R = 3点9点6点8点から3点O点3点8点を引いたものです 温度から0点2点4点相対湿度を引いたものです これはこの方程式です これは実用的に重要です 
さて、ここで少し裏ワザをしてみましょう エクセルに入力してみましょう 
温度が10℃で湿度が40％の場所があったとします この式が正しいとすると Rは約2ポイント7になります 温度が35℃で湿度が80％の場所では 約8ポイントになります 


実質的に重要なことなの？ああ、そうです、そうですね？
異なる気候を持つ２つの異なる都市は、他のすべての点で同じで、このモデルが正しければ、１つの都市では病気が広がらず（Rが1より小さいので）、１つの都市では大規模な指数関数的爆発が起こるでしょう。
ですから、もしこのモデルが正しければ、これは非常に実用的に重要な結果であることがわかります。
このように、モデルの実用的な重要性を判断する方法は、p値ではなく、実際の結果を見て判断することです。
ですから、モデルの実用的な重要性をどのように考え、予測モデルをどのようにして生産に役立つものに変えていくかということです。
そこで、私はこのことについて何年も考えていましたが、実際に他の素晴らしい人たちと一緒に論文を作成しました。
"優れたデータ製品の設計 
これは私が設立したオプティマル・ディシジョン・グループという会社で 10年に渡って行ってきた仕事に 基づいています 
オプティマル・ディシジョンズ・グループは、保険会社がどのような価格を設定すべきかを把握するのを支援することに重点を置いていました。
それまでの保険会社は、予測モデリングに重点を置いていました。
特にアクチュアリーは、自分の車を衝突させる可能性がどの程度あるのか、また衝突した場合の損害はどの程度なのかを把握し、それに基づいて保険会社がどのような価格を設定すべきかを考えることに時間を費やしていました。
この会社の場合は、別のアプローチを使うことにしました。
保険の例で言うと、保険会社の目的は、いかにして5年間の利益を最大化するかということです。
そして、どのようなインプットをコントロールすることができるかということです。
そして、データはデータであり、レバーを変えることで目的がどう変わるかを知ることができます。
例えば、車を壊す可能性の高い人たちに価格を上げれば、その人たちの数は減り、コストは減りますが、同時に収入も減ることになります。
そこで、収集したデータを介して、目的とレバーを結びつけるために、レバーがどのように目的に影響を与えるかを説明するモデルを構築しました。
このように言うと、当たり前のことのように思えますが、私たちが1999年にOptimal Decisionsの仕事を始めたときには、保険業界では誰もこのようなことをしていませんでした。
保険業界では誰もが予測モデルを使って 車を衝突させる可能性を推測していました そして価格設定は20％とか何かを追加して設定されていました
それは非常に素朴な方法で行われていました。


私がしたことは、この基本的なプロセスを何年にもわたって、多くの企業が予測モデルを行動に移すために、このプロセスをどのように使うかを理解する手助けをしてきたことです。
特定のモデルで実際に価値を得るための出発点は、自分が何をしようとしているのかを考えることであり、自分がしようとしていることの価値の源泉は何かを知ることです。
レバー - 何を変えることができるのか？
何もできないのであれば、予測モデルの意味があるのではないでしょうか？
自分が持っていないデータを見つける方法を見つけ出し、適切なデータ、利用可能なデータを見つけ出し、その上でどのようなアプローチでアナリティクスを行うかを考えます。
そして超重要なのは、その変更を実際に実行できるかどうかです。
そして、環境の変化に合わせて実際にどのように変化させていくかということも非常に重要です。
興味深いことに、これらのことの多くは、学術的な研究があまり行われていない分野です。
少しだけですが。
特に「メンテナンス」に関する論文の中には、機械学習モデルがまだ大丈夫かどうかをどうやって判断するのか、時間をかけてどうやって更新するのか、などがあります。
どうやって時間をかけて更新するのか？
これまでにもたくさんの引用がありましたが、多くの人が数学に集中しているため、あまり頻繁には出てきませんでした。
そうですね 
そして、「この全体にはどんな制約があるのか」という全体的な質問があります。この本の中には、この６つのことを一つ一つ解説した付録がついています。
そして、例のリストがあります。
これは、価値について考える方法の一例です。
また、企業や組織が考えてみるために使える質問もたくさんあります。
質問があります。
はい、ちょっと待ってください。
私が言おうとしていたのは、この付録をチェックしてみてください。なぜなら、この付録は元々はブログ記事として書かれていたからです。


これは何十万回も再生されています。
それは、機械学習からどのように価値を得るのか、実践するのか、そして実際に何を求めているのかについての20年間の苦労して勝ち取った洞察のようなものです。
ぜひチェックしてみてください。
季節性とcovid-19の透過性の関係について人々はどのように考えるべきか、という質問について考えるときには、データの中の数字が何なのかだけでなく、実際にはどのように見えるのか、という質問を深く掘り下げる必要があります。
論文の中で、彼らが示しているものの一つは、実際の地図です 温度と湿度とRの右の地図です 
驚くことではありませんが、中国の湿度と気温は、私たちが自動相関と呼んでいるものです。
つまり、地理的に近い場所では、気温も湿度も似ています。
このように、これは実際に、彼らが持っているp値を問題にしています 
なぜなら、これらの都市を100個の全く別の都市とは考えられないからです 
互いに近接している都市は、おそらく非常に近い挙動をしていますので、これらの都市は少数の都市の集合体のようなものと考えるべきかもしれません。
このように、実際にモデルを検討する際には、何が限界なのか、何が限界なのかを考える必要があります。 
そして、それが何を意味するのか？
それをどうすればいいのか？
このような効用の観点から考える必要があります。
結果の順序はどうなっているのか？ 
ただの帰無仮説検定ではありません。
このケースでは、基本的に4つの主要な方法が考えられます。
温度とRの間には実際に関係があるということになるかもしれません。
あるいは、温度とRの間には実際には関係がないということです。
そして、関係があると仮定して行動するかもしれません。


あるいは、関係がないと仮定して行動するかもしれません。
だから、４つの可能性をそれぞれ見て、経済的、社会的な結果はどうなるのか？ 
失われた命には大きな違いがありますし、経済の崩壊やその他何でもあります。
この論文では、もし彼らのモデルが正しければ、3月の世界の全ての都市のR値はどうなるかを示しています。
そして、世界の全ての都市の7月のR値の可能性が高いことが示されています。
例えば、ニューイングランドやニューヨークを見てみましょう ここでの予測では、西海岸や西海岸の海岸線では 7月に病気が広がらなくなると予測されています 
もしそうなったら、もし彼らの予測が正しければ、それは大惨事になります。
「この病気は問題ではない事が判明した」と言うでしょう。
"科学者は間違っていた" 人々 は以前の日常生活に戻るし、我々 は 1918 年に起こったことを見ることができますインフルエンザ ウイルスのような 2 回目の周りを行く。
冬が来た時には、最初よりもずっと悪い状態になるかもしれません。
これが本当か嘘かによって、政策に大きな影響を与える可能性があります。
考えてみましょう 
いいですか？私が言いたかったのは、「夏になれば解決する」と考えるのは非常に無責任だということです。
今すぐに行動する必要はありません"  ただ、これは指数関数的に成長しているものであり、膨大な被害をもたらす可能性があるということだけです。
ええ、はい、わかりました。
すでにそうなっている 
もし季節性があると仮定して、夏に解決すると仮定したら、今は無関心になるかもしれない。
もし季節性がないと仮定して、季節性があると仮定した場合、実際に起こる破壊への期待が大きくなり、あなたの人口はさらに無関心になるでしょう。
どの方向から見ても間違っていることが問題になります。
ですから、この種のモデリングを行う際の対処法の1つは、前置詞について考えてみることです。


推定値とは、基本的には、帰無仮説を持つのではなく、何がより可能性が高いのかを推測することです。 
この場合、記憶が正しければ、インフルエンザウイルスが27℃で不活性化するように、風邪のように、風邪のコロナウイルスは季節性があることがわかっています。
1918年のインフルエンザの大流行は 季節性でした 
これまでに研究された全ての国や都市でこのような研究が行われてきました 
これまでの研究では常に気候との関係を発見しています 
だから私たちはこう言うのかもしれません "これはおそらく季節的なものだと 思われています" それで私たちはこう言うの "この論文は、それにいくつかの証拠を加えている" このように、モデルを実際に使用して、この場合の政策議論だけでなく、組織の意思決定にも使用することがいかに複雑であるかを示しています。
なぜなら、ご存じのように、常に複雑さや不確実性があるからです。
そのため、実際にはユーティリティについて考えなければなりません。
そして、最善の推測をして、可能な限りすべてのものを組み合わせてみてください。
分かったわ 
以上のことを言ったが 
予測モデルであっても、それ自体が有用な場合もありますが、モデルを稼働させることができるのは素晴らしいことです。
何かをプロトタイプ化するのに役立つこともありますし、何か大きな絵の一部でなければならないこともあります。
だから、ここでは巨大なエンドツーエンドのモデルを作ろうとするのではなく、どのようにして予測モデルを作成するのかをお見せしようと考えました。
Pytorch FastAIモデルを起動して実行する方法を紹介しようと思った。
可能な限り生の状態で。
そこから、好きなように、その上に構築することができます。
そのために、我々は独自のデータセットをダウンロードし、キュレーションする。
そして、あなたも同じことをします。
そのデータセットを使って自分のモデルを学習し、アプリケーションを作成して、それをホストするのです。


そうでしょう? さて，画像データセットを作成する方法はたくさんあります．自分のパソコンに写真があるかもしれませんし，職場にあるものを使うかもしれません．
しかし、最も簡単な方法は、インターネットから画像をダウンロードすることです。
インターネットからダウンロードできるサービスはたくさんあります。
ここではBing Image Searchを使います。
これらのサービスはとても使いやすいからです。
他のサービスは利用規約を破る必要があります。
そのため、ここではその方法をお見せするつもりはありません。
しかし、その方法を示す例はたくさんあります。
あなたもチェックしてみてください。
Bingの画像検索は、今のところとても良いです。
今後のアップデートでおすすめ度が変わることもあるので、ウェブサイトで確認してみてください。
Bing Image Searchの最大の問題点は、登録手続きが悪夢のようなものだということです。
本書の中で最も困難な部分の1つは、彼らのクソAPIにサインアップするだけです。
それにはAzureを経由する必要があります。
それは Cognitive Services - Azure Cognitive Services と呼ばれています。
そのため、サインアップの方法については、すべての情報がウェブサイトに掲載されていることを確認しておきましょう。
ということで、すでにサインアップされているという前提で話を進めていきます。
でも、探せば出てきますよ。Bing, Bing Image Search APIです。
今のところ、7日間無料で利用できます。
その後は何度でも利用できますが、1秒間に3回のトランザクションに制限されています。


どっちがまだ十分だよ。
無料でも数千件の検索ができるので、今のところは無料でもかなりいい感じです。
Bing画像検索などのサービスに登録すると、APIキーが発行されます。
ここにある「XXX」をAPIキーに置き換えてください。
これを「キー」と呼びます。
実際には、ここでやってみましょう。
キーを入力して、search_images_bingという関数を作成しました。
ご覧のように、たった2行のコードですが、少しでも時間を節約しようと思って、APIキーと検索語を入力して、その検索語にマッチするURLのリストを返します。
この特定のサービスを利用するには特定のパッケージをインストールする必要があります。
そうすれば、これを実行することができるようになり、デフォルトで150のURLが返ってくると思います。
さて、fast.aiにはdownload_url関数がついていますので、画像をダウンロードして確認してみましょう。
それで何をしたかというと、"grizzly bear "で検索したら、グリズリーベアが出てきました。
そこで私が言ったのは、グリズリーベアとツキノワグマとテディベアを 認識できるモデルを作ってみようと言うことです。
キャンプ場の近くにビデオ認識システムを設置してクマの警告を表示しますが、もしテディベアが来ても警告してくれませんし、私を起こしてくれません。
そこで、この３種類のクマをそれぞれ調べて、グリズリー、クロクマ、テディベアという名前のディレクトリを作り、Bingでクマと一緒に特定の検索語で検索してダウンロードします。
download_imagesもfast.ai関数です。
この後にget_image_filesを呼び出すことができます。これはfast.ai関数で、このパス内のすべての画像ファイルを再帰的に返してくれます。
このパスの中にあるすべての画像ファイルを再帰的に返してくれます。
注意しなければならないことの一つは、ダウンロードしたものの多くが画像ではないものになってしまい、壊れてしまうということです。
そこで verify_images をコールして、これらのファイル名がすべて実際の画像であるかどうかを確認します。


で、今回の場合は失敗したものがなかったので、空になっています。
しかし、もしあったのであれば、Path.unlinkを呼び出してリンクを解除します。
Path.unlinkはPython標準ライブラリの一部で、ファイルを削除してくれます。
そして map は、このコレクションの要素ごとにこの関数を呼び出してくれるものです。
これは "L "と呼ばれる特別なfast.aiクラスの一部です。
基本的にはPython標準ライブラリのリストクラスとnumpy配列クラスをミックスしたようなものです。
このコースでは後ほど詳しく説明しますが、基本的にはPythonでより関数型のプログラミングが簡単にできるようにしようとしています。
この場合、失敗したリストにあるすべての画像のリンクを解除します。
これで、たくさんの画像が含まれたパスができました。
ということで、モデルを作成します。
モデルを作成するにはまず、fast.aiにどのようなデータがあり、どのように構造化されているかを伝える必要があります。
レッスン1ではファクトリーメソッドと呼ばれるものを使ってモデルを作成しました。
これらのファクトリメソッドは初心者には良いのですが、これからレッスン２に入ります。
ここからは、データを好きな形式で使うための超柔軟な方法を紹介します。
DataBlock APIは以下のようになっています。
これがDataBlock APIです。
fast.aiに独立変数と従属変数を指定します。
ラベルと入力データを指定します。
この場合の入力データは画像で、ラベルはカテゴリです。
カテゴリーはグリズリーか黒かテディになります。


これが最初に伝えることですね。
これがブロックのパラメータです。
そして、それを指定します - この場合はファイル名のリストを取得するにはどうすればいいのでしょうか？
この関数を自分で呼んでみたので、その方法がわかりました。
関数は get_image_files と呼ばれています。
このリストを取得するためにどのような関数を使うかを指定して、データを検証セットと学習セットにどうやって分けるかを指定します。
そこで、RandomSplitterと呼ばれるものを使って、データをランダムに分割します。
そして、そのうちの30%を検証セットに振り分けます。
また、ランダムシードを設定して、これを実行するたびに検証セットが同じになるようにします。
そして、データにラベルを付けるにはどうすればいいかと言うと、これは関数の名前です。
これはparent_labelという関数の名前です。
これは親の名前で各項目を探します。
これは、この特別なものは黒熊になります 
これは画像データセットを表現するための最も一般的な方法です。
そして最後にitem_tfmsと呼ばれるものがあります。
変換についてはもう少し詳しく説明します。
これらは基本的に各画像に適用される関数です。
それぞれの画像は128×128の正方形にリサイズされます。
DataBlock APIについては近日中に詳しく説明する予定です。
しかし、基本的には、画像ファイルのリストである get_items を呼び出すことになります。


そして get_x, get_y を呼び出します。この場合は get_x はありませんが、get_y はありますので、親ラベルだけです。
そして、これらの2つのことについてそれぞれcreateメソッドを呼び出します - 画像を作成し、カテゴリを作成します。
そして、item_tfmsを呼び出してサイズを変更します。
次にそれをデータローダと呼ばれるものに入れます。
データローダとは、一度に数枚の画像（デフォルトでは64枚だと思います）を取得して、それらを一つのバッチにまとめるものです。
これはバッチと呼ばれていて、64枚の画像を取得して、それらをすべて一つにまとめます。
なぜそうするかというと、すべての画像を一度にGPU上に配置して、GPUを介してモデルに一度にすべての画像を渡すことができるようにするためです。
これによりGPUの処理速度が格段に向上します。
そして最後に（ここでは何も使いませんが）、バッチ変換と呼ばれるものができます。
そして、ここでは概念的にはスプリッタと呼ばれるものがありますが、これは学習セットと検証セットに分割するものです。
これはfast.aiにデータをどのように扱うかを指示するための非常に柔軟な方法です。
そして最後にDataLoaders型のオブジェクトを返します。
これをいつもDLと呼ぶのはそのためですね。
DataLoadersには検証用とトレーニング用のDataLoaderがあります。
先ほど言ったデータローダとは、いくつかの項目のバッチを一度に取得してGPU上に配置するものです。
これが基本的にデータローダのコード全体です。
細かいことは重要ではありませんが、fast.aiの多くの概念のように、実際に見てみると、信じられないほどシンプルで小さなものです。
文字通り、いくつかのデータローダを渡すだけで、属性に保存されます。
パスを渡すと、最初のデータは.train、2番目のデータは.validとして返されます。
つまり、まずDataBlockを作成してDataLoadersを呼び出し、DLを作成するためのパスを渡すことでDataLoadersを作成することができます。


そして、その上でshow_batchを呼び出すことができます。
fast.aiではshow_batchをかなり何でも呼び出せるので、データを見ることができます。
見てください、グリズリーがいます、テディがいます、グリズリーがいます。
というわけで、その通りになりました。
来週はデータ拡張を見ようと思ってるからデータ拡張は飛ばしてモデルのトレードに飛び込もう 
さて、DLができたら、レッスン１と同じように cnn_learner を呼び出して ResNet を作成します。
今回は小さめの ResNet18 を作成します。
ここでもエラー率を求めて、.fine_tune をもう一度呼びます。
これまでに見てきたコードと同じ行を使っています。
誤差率が９から１に下がっているのがわかります。１％の誤差で、約２５秒のトレーニングの後に 
1分もかからずに学習して、たった450枚の画像を取得したことがわかります。
ご覧のように、"実際に黒クマであるものについて、黒クマ対グリズリーベア対テディベアの予測はどれくらいでしょうか？"というようなものです。 
で、対角線上にあるのが全部当たっているので、2つの誤差が出ているように見えます。
黒だと予測されていたグリズリーが１匹とグリズリーだと予測されていた黒が１匹です 
超、超、超、便利な方法が「トップロスのプロット」ですこれは実際に私の誤差がどのように見えるかを教えてくれます 
これはグリズリーだと予測されていましたがラベルは "ツキノワグマ "でした 
これはツキノワグマと予測されていたがラベルは「グリズリーベア」だった 
こちらのものは実は間違っていません。
これは「黒」と予測されていて、実際には黒なのです。
しかし、この中に出てくるのは、これらがモデルが最も自信を持っていなかったものだからです。


さて、来週は画像分類器のクリーナーについて見ていきましょう。
ここでは、これをどうやって本番させるかに焦点を当ててみましょう。
本番環境に導入するためには、モデルをエクスポートする必要があります。
モデルをエクスポートすると、新しいファイルが作成されます。デフォルトでは "export.pkl "という名前で、モデルのアーキテクチャとすべてのパラメータが含まれています。
これで、どこかのサーバーにコピーして、定義済みのプログラムとして扱うことができるようになりました。
このようにして、訓練されたモデルを新しいデータ上で使用するプロセスを「推論」と呼びます。
ここでは、学習者を再びロードして推論学習者を作成していますが、明らかにノートに保存した後にすぐ隣で行うのは意味がありません。
しかし、これはどのように機能するかをお見せしているだけです。
これはサーバー上で行う推論です。
一度モデルを学習したら、プログラムとして扱うことができることを覚えておいてください。
これが私たちのプログラムです。
これが私たちの熊の予測器です。
これで "predict "を呼び出して画像を渡すと、99.999%の確率で "グリズリー "であることを教えてくれます。
来週は熊分類器の実際のGUIを作成して終わりにしましょう。
"Binder "というサービスを使って 無料で実行する方法を紹介します それから、裏で何が起こっているのかの詳細に 飛び込む準備ができていると思います 
質問や他に何かありますか レイチェル？ ないわ 
オーケー、それでいい 
みんなありがとう 
ここまでで、機械学習の観点からの基礎となる重要な部分はほとんどカバーできたと思います。
では、ディープラーニングがどのように裏で機能しているのか、低レベルの詳細に飛び込んでいく準備ができているので、それは来週から始まると思います。
それでは、その時にお会いしましょう。
