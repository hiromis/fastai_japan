Quindi, ciao a tutti e bentornati a Practical Deep Learning for Coders. 
Questa è la seconda lezione, e nell'ultima lezione abbiamo iniziato ad allenare i nostri primi modelli. 
Non avevamo la minima idea di come quella formazione stesse realmente funzionando, ma stavamo guardando ad un livello più alto di quello che stava succedendo. 
E abbiamo imparato "Cos'è l'apprendimento meccanico?" e "Come funziona?" e ci siamo resi conto che, in base a come funzionava l'apprendimento meccanico, ci sono alcune limitazioni fondamentali su ciò che può fare, e abbiamo parlato di alcune di queste limitazioni. 
E abbiamo anche parlato di come, dopo aver addestrato un modello di machine learning, ci si ritrova con un programma che si comporta come un normale programma o qualcosa del genere: con degli input e una cosa in mezzo e degli output. 
Quindi oggi finiremo di parlare di questo e poi vedremo come far entrare in produzione quei modelli e quali potrebbero essere alcuni dei problemi che si presentano nel farlo. 
Volevo ricordarvi che ci sono due serie di libri - mi spiace, due serie di quaderni - a vostra disposizione. 
Uno è il fastbook repo (i quaderni veri e propri contenenti tutti i testi del libro di O'Reilly) e quindi questo vi permette di vedere tutto quello che vi sto dicendo in modo molto più dettagliato, e poi oltre a questo c'è il corso v4 repo che contiene esattamente gli stessi quaderni ma con tutta la prosa strappata per aiutarvi a studiare. 
Quindi è lì che vuoi davvero fare il tuo esperimento e la tua pratica e così forse mentre ascolti il video puoi tipo passare avanti e indietro tra il video e la lettura o fare uno e poi l'altro, e poi metterlo via e dare un'occhiata ai quaderni del corso v4 e cercare di ricordare come "Ok, di cosa trattava questa sezione?" ed eseguire il codice, e vedere cosa succede e cambiarlo e così via. 
Così abbiamo guardato questa linea di codice dove abbiamo visto come abbiamo creato i nostri dati passandoci le informazioni - forse soprattutto un modo per etichettare i dati - e abbiamo parlato dell'importanza dell'etichettatura. 
E in questo caso, questo particolare insieme di dati, sia che si tratti di un gatto o di un cane, si può capire se si tratta di una lettera maiuscola o di una lettera minuscola nella prima posizione. 
È proprio così che funziona questo dataset (che ti dicono quando il readme). 
E abbiamo anche guardato in particolare a questa idea di "percentuale valida uguale a 0,2", e come "Cosa significa? Crea un set di validazione", ed era qualcosa di cui volevo parlare di più. 
La prima cosa che voglio fare, però, è sottolineare che questa particolare funzione di etichettatura restituisce qualcosa che è vero o falso. 
E in realtà questo set di dati, come vedremo più avanti, contiene anche la razza effettiva di 37 diverse razze di cani e gatti, quindi si può anche prendere questo dal nome del file. 
In ognuno di questi due casi stiamo cercando di prevedere una categoria "È un gatto, o è un cane?" o "È un pastore tedesco, o un Beagle, o un gatto Ragdoll, o qualsiasi altra cosa? Quando si cerca di prevedere una categoria, quindi quando l'etichetta è una categoria, la chiamiamo modello di classificazione. 
D'altra parte, si potrebbe cercare di prevedere quanto è vecchio l'animale, o quanto è alto, o qualcosa del genere, che è come un numero continuo che potrebbe essere come 13,2 o 26,5 o qualsiasi altra cosa. 
Ogni volta che si cerca di prevedere un numero, l'etichetta è un numero che si chiama regressione. 
Va bene? Quindi questi sono i due tipi principali di classificazione del modello e regressioni. 
Questo è un gergo molto importante da conoscere. 


Quindi il modello di regressione tenta di prevedere una o più quantità numeriche come la temperatura, o la posizione, o qualsiasi altra cosa. 
Questo è un po' confuso, perché a volte le persone usano la parola regressione come scorciatoia per un particolare, per un... Come un'abbreviazione per un particolare tipo di modello, chiamato regressione lineare. 
Questo è super confuso, perché non è questo il significato della regressione. 
La regressione lineare è solo un particolare tipo di regressione, ma volevo solo avvertirvi di questo. 
Quando si inizia a parlare di regressione molte persone penseranno che si tratti di regressione lineare, anche se non è questo il significato della parola. 
Va bene, volevo parlare di questa cosa valida per cento zero virgola due. 
Così come abbiamo descritto la percentuale valida afferra, in questo caso, il venti per cento dei dati, se è zero virgola due, e li mette da parte come in un secchio a parte e poi quando alleni il tuo modello, il tuo modello non riesce a guardare affatto quei dati. 
Quei dati servono solo per decidere, per mostrarvi quanto è preciso il vostro modello. 
Quindi, se vi allenate troppo a lungo, e o con pochi dati, e/o un modello con troppi parametri, dopo un po' di tempo la precisione del vostro modello peggiorerà, e questo si chiama "overfitting". 
Giusto? Quindi usiamo il set di validazione per assicurarci di non essere in overfitting. 
La prossima linea di codice che abbiamo guardato è questa, dove abbiamo creato qualcosa chiamato "learner". 
Impareremo molto di più su questo, ma uno studente è fondamentalmente, o è, qualcosa che contiene i vostri dati e la vostra architettura che è la funzione matematica che state ottimizzando, e quindi uno studente è la cosa che cerca di capire quali sono i parametri che meglio fanno sì che questa funzione corrisponda alle etichette di questi dati. 
Quindi parleremo molto di più di questo, ma fondamentalmente questa particolare funzione ResNet34 è il nome di una particolare architettura che è semplicemente molto buona per i problemi di visione artificiale. 
In realtà il nome è davvero ResNet e poi 34 vi dice quanti strati ci sono. 
Quindi si possono usare quelli con numeri più grandi qui per ottenere più parametri che richiederanno di allenarsi, prendere più memoria, più probabile che si sovrapponga, ma potrebbe anche creare modelli più complessi. 
In questo momento però volevo concentrarmi su questa parte qui che è metrica uguale a tasso di errore. 
Qui è dove si elencano le funzioni che si vogliono... 
Che vuoi essere chiamato con i tuoi dati. 
Con i vostri dati di validazione e stamparli dopo ogni epoca, e l'epoca è ciò che noi la chiamiamo quando si guarda ogni singola immagine nel set di dati una volta. 
E così dopo che avete guardato ogni immagine nel set di dati una volta che abbiamo stampato alcune informazioni su come state facendo e la cosa più importante che stampiamo è il risultato della chiamata di queste metriche, quindi il tasso di errore è il nome di una metrica ed è una funzione che stampa solo ciò che la percentuale del set di validazione è stata erroneamente classificata dal vostro modello. 


Quindi la nostra metrica è una funzione che misura la qualità delle previsioni utilizzando il set di validazione, per cui i tassi di errore sono comuni e la precisione è solo 1 meno il tasso di errore, così importante da ricordare che la scorsa settimana abbiamo parlato di perdita. 
Arthur Samuel ha avuto questa importante idea nell'apprendimento delle macchine che abbiamo bisogno di un modo per capire quanto sia buono il nostro modello, in modo che quando cambiamo i parametri possiamo capire quale set di parametri fa sì che la misurazione delle prestazioni migliori o peggiori, che la misurazione delle prestazioni sia chiamata perdita. 
La perdita non è necessariamente la stessa della vostra metrica. 
Il motivo è un po' sottile e lo vedremo in modo molto dettagliato una volta che avremo approfondito la matematica nelle prossime lezioni, ma fondamentalmente è necessaria una funzione in cui se si modificano i parametri solo di un po' verso l'alto o solo di un po' verso il basso si può vedere se la perdita migliora o peggiora e si scopre che il tasso di errore e l'accuratezza non lo dice affatto perché si potrebbero cambiare i parametri di una quantità così piccola che nessuna delle previsioni del vostro cane inizia a diventare un gatto e nessuna delle previsioni del vostro gatto inizia a diventare un cane. 
Quindi, come le vostre previsioni non cambiano, così il vostro tasso di errore non cambia. 
La perdita e la metrica sono strettamente correlate, ma la metrica è la cosa che vi interessa della perdita è quella che il vostro computer sta usando come misura delle prestazioni per decidere come aggiornare i vostri parametri. 
Quindi misuriamo l'overfitting guardando le metriche sul set di validazione. 
Quindi l'IA veloce usa sempre il set di validazione per stampare le vostre metriche e l'overfitting è come la cosa fondamentale che l'apprendimento a macchina riguarda è come trovare un modello che si adatti ai dati non solo per i dati con cui ci stiamo allenando, ma per i dati che l'algoritmo di allenamento non ha mai visto prima. 
Quindi i risultati si adattano eccessivamente quando il nostro modello è fondamentalmente "imbrogliare". 
Un modello può imbrogliare dicendo oh ho già visto questa immagine esatta e ricordo che è l'immagine di un gatto. 
Quindi potrebbe non aver imparato come sono fatti i gatti in generale, ma ricorda solo che le immagini uno quattro e otto sono gatti e due e tre e cinque sono cani e non impara nulla sul loro aspetto reale. 
Questo è il tipo di imbroglio che cerchiamo di evitare, non vogliamo che memorizzi il nostro particolare set di dati. 
Quindi ci dividiamo i nostri dati di convalida e la maggior parte di queste parole che si vedono sullo schermo provengono dal libro, quindi le ho copiate e incollate. 
Quindi, se dividiamo i dati di validazione e ci assicuriamo che il nostro modello non li veda mai durante l'addestramento, non ne rimane traccia, quindi non possiamo assolutamente imbrogliare. 
Non è del tutto vero! Possiamo imbrogliare, il modo in cui potremmo imbrogliare è che potremmo correre, potremmo adattare un modello al risultato e il set di validazione, cambiare qualcosa di un po' più adatto ad un altro modello, guardare il set di validazione, cambiare qualcosa di un po' più adatto ad un altro modello, guardare il set di validazione, cambiare qualcosa di un po' più adatto a un altro modello, cambiare qualcosa di un po' più adatto a un altro modello, potremmo farlo un centinaio di volte, fino a quando non troviamo qualcosa con il set di validazione che sia il migliore. 
Ma ora potremmo aver adattato il set di validazione, giusto? 
Quindi, se volete essere davvero rigorosi su questo, dovreste mettere da parte un terzo bit di dati chiamato set di test che non viene utilizzato per la formazione e non viene utilizzato per le vostre metriche. 
In realtà, non lo si guarda fino a quando l'intero progetto non è finito. 
E questo è ciò che viene utilizzato su piattaforme da competizione come Kaggle. 
Su Kaggle, dopo la fine della competizione le vostre prestazioni saranno misurate rispetto a un set di dati che non avete mai visto. 



E quindi, questo è un approccio davvero utile e in realtà è un'ottima idea farlo anche se non si sta facendo il modello da soli. 
Quindi, se state guardando i venditori e state solo cercando di decidere oggi di andare con IBM o Google o Microsoft e tutti vi stanno mostrando quanto sono grandi i loro modelli, quello che dovreste fare è dire: "Ok, andate a costruire i vostri modelli e io mi aggrapperò al 10% dei miei dati e non ve li farò vedere affatto e quando avrete finito, tornate e poi farò girare il vostro modello sul 10% dei dati che non avete mai visto". 
Ora, però, tirare fuori i tuoi set di validazione e di test è un po' sottile. 
Ecco un esempio di un semplice piccolo set di dati e questo viene da un fantastico post sul blog che Rachel ha scritto e al quale ci collegheremo per creare dei set di validazione efficaci. 
E potete vedere che in pratica avete una sorta di set di dati stagionali. 
Ora, se solo diceste: "Ok, fas.ai, voglio modellare che voglio creare un mio dataloader usando un valid_percento di 0.2", farebbe questo. 
Cancellerebbe in modo casuale alcuni punti, giusto? 
Ora, questo non è molto utile perché possiamo ancora barare perché questi punti sono proprio in mezzo ad altri punti e questo non è quello che accadrebbe in pratica. 
Quello che accadrebbe in pratica è che vorremmo predire le vendite per data, a destra vogliamo predire le vendite per la prossima settimana. 
Non i saldi di 14 giorni fa 18 giorni fa e 29 giorni fa, ok? 
Quindi quello che dovete fare per creare un set di validazione efficace non è farlo in modo casuale, ma invece tagliare la fine, giusto? 
E così questo è quello che succede in tutte le competizioni Kaggle che richiedono tempo, per esempio, è la cosa che devi prevedere è la prossima, tipo due settimane o giù di lì dopo l'ultimo punto dati che ti danno e questo è quello che dovresti fare anche per il tuo set di test, quindi se hai dei venditori che stai guardando dovresti dire loro che va bene dopo che hai finito di modellare, controlleremo il tuo modello rispetto ai dati che sono una settimana più tardi di quanto tu abbia mai visto prima. 
E non sarete in grado di riqualificarvi o altro perché questo è quello che succede nella pratica, giusto? Ok... 
C'è una domanda, ho sentito dire che le persone descrivono l'overfitting come un errore di formazione che è al di sotto dell'errore di validazione, questa regola empirica finisce per essere all'incirca la stessa della tua? 
Ok, questa è un'ottima domanda. 
Quindi, penso che quello che intendono sia la perdita di allenamento contro la perdita di validazione. 
Poiché non stampiamo l'errore di addestramento, alla fine di ogni epoca stampiamo il valore della vostra funzione di perdita per il set di addestramento e il valore della funzione di perdita per il set di validazione. 
E se vi allenate abbastanza a lungo, è così, se si tratta di allenamenti, la perdita di allenamento si riduce e la perdita di validazione si riduce. 
Perché per definizione, la funzione di perdita è definita come una funzione di perdita inferiore è un modello migliore. 
Se iniziate ad allenarvi troppo, la vostra perdita di allenamento continuerà a diminuire, giusto? 


Perché perché non dovrebbe? 
Sai, i parametri sono sempre migliori. 
Ma la tua perdita di convalida inizierà a salire perché in realtà hai iniziato ad adattarti ai punti dati specifici del set di allenamento e quindi non migliorerà. 
Non migliorerà per il set di validazione, ma inizierà a peggiorare. 
Tuttavia, questo non significa necessariamente che si sta sovraadattando o almeno che non si sta sovraadattando in modo sbagliato, perché vedremo che è effettivamente possibile essere in un punto in cui la perdita di validazione sta peggiorando, ma l'accuratezza o l'errore di validazione o la metrica sta ancora migliorando. 
Quindi non descriverò ancora matematicamente come ciò accadrebbe, perché dobbiamo imparare di più sulle funzioni di perdita, ma lo faremo. 
Ma per ora basta capire che la cosa importante da guardare è che la vostra metrica sta peggiorando, non la vostra funzione di perdita sta peggiorando. 
Grazie per questa fantastica domanda. 
La prossima cosa importante che dobbiamo imparare si chiama apprendimento del trasferimento. 
Quindi la prossima riga di codice diceva learn.fine_tune. 
Perché dice learn.fine_tune? 
Fine tune è ciò che facciamo quando trasferiamo l'apprendimento, quindi il transfer learning è l'utilizzo di un modello pre-formato per un compito diverso da quello per cui è stato originariamente addestrato. 
Quindi più gergo per capire il nostro gergo. 
Diamo un'occhiata a questo. 
Cos'è un modello pre-formato? 
Quindi cosa succede se vi ricordo che vi ho detto che l'architettura che stiamo usando si chiama ResNet-34? 
Quindi quando prendiamo quel ResNet-34 è solo una funzione matematica, con molti parametri che andranno bene per l'apprendimento automatico. 
C'è un grande set di dati chiamato ImageNet, che contiene 1,3 milioni di immagini di mille tipi diversi di cose, che si tratti di funghi, animali, aerei, martelli o altro. 
C'è un concorso o c'è una gara che si svolge ogni anno per vedere chi riesce ad ottenere la migliore precisione sul concorso ImageNet. 
E i modelli che andavano veramente bene, la gente prendeva quei valori specifici di quei parametri e li rendeva disponibili su internet per chiunque potesse scaricarli. 


Quindi, se scaricate che non avete solo un'architettura ora avete un modello addestrato. 
Avete un modello in grado di riconoscere mille categorie di cose in immagini. 
Il che probabilmente non è molto utile a meno che non si desideri qualcosa che riconosca esattamente quelle mille categorie di cose. 
Ma si scopre che si può piuttosto iniziare con quei pesi nel vostro modello e poi addestrare qualche altra epoca sui vostri dati e si finirà con un modello molto più preciso di quello che si otterrebbe se non si iniziasse con quel modello pre-formato e vedremo perché tra un attimo, giusto?
 Ma questa idea di trasferire l'apprendimento, è un po'... ha un senso intuitivo, no? 
ImageNet ha già alcuni gatti e alcuni cani e si sa che può dire che questo è un gatto e questo è un cane, ma si vuole forse fare qualcosa che riconosce molte razze che non sono in ImageNet. 
Beh, per poter riconoscere i gatti contro i cani contro gli aeroplani contro i martelli deve capire cose come: che aspetto ha il metallo? 
Che aspetto ha la pelliccia? 
Che aspetto hanno le orecchie? 
Sapete, così può dire come oh questa razza di animali, questa razza di cani ha le orecchie a punta e oh questa cosa è di metallo quindi non può essere un cane. 
Quindi tutti questi tipi di concetti vengono appresi implicitamente da un modello preaddestrato. 
Quindi, se si inizia con un modello preaddestrato, non è necessario imparare tutte queste caratteristiche da zero, e quindi trasferire l'apprendimento è la cosa più importante per poter utilizzare meno dati e meno calcoli e ottenere una migliore precisione. 
Quindi questo è un punto chiave per la libreria dei fastai e un punto chiave per questo corso. 
C'è una domanda: Sono un po' confuso sulle differenze tra perdita, errore e metrica. 
Certo, quindi l'errore è solo un tipo di metrica, quindi ci sono un sacco di possibili etichette diverse che si potrebbero avere. 
Diciamo che stavi cercando di creare un modello che potesse prevedere l'età di un gatto o di un cane. 
Quindi la metrica che potresti usare è: in media, di quanti anni sei stato fuori? 
Quindi sarebbe una metrica. 
D'altra parte se stai cercando di prevedere se si tratta di un gatto o di un cane la tua metrica sarebbe: in che percentuale del tempo mi sbaglio? 
Quindi quest'ultima metrica si chiama tasso di errore. 


Ok, quindi l'errore è una metrica particolare. 
È una cosa che misura quanto bene stai facendo ed è come se dovesse essere la cosa a cui tieni di più. 
Quindi si scrive una funzione o si usa una di quelle predefinite del fastai che misura quanto bene si sta facendo. 
La perdita è la cosa di cui abbiamo parlato nella Lezione Uno, quindi farò un breve riassunto, ma tornate alla lezione uno se non ve lo ricordate. 
Arthur Samuel ha parlato di come un modello di machine learning abbia bisogno di una qualche misura di performance che possiamo guardare: quando regoliamo i nostri parametri in alto o in basso questa misura di performance migliora o peggiora? 
E come ho detto prima, alcune metriche non cambiano affatto se si spostano i parametri su e giù solo un po'. 
Quindi non possono essere usati per questo scopo di regolare i parametri per trovare una migliore misura delle prestazioni. 
Quindi molto spesso abbiamo bisogno di usare una funzione diversa che chiamiamo funzione di perdita e la funzione di perdita è la misura delle prestazioni che l'algoritmo usa per cercare di migliorare i parametri ed è qualcosa che dovrebbe essere abbastanza vicino alla metrica a cui si tiene, ma è qualcosa che, cambiando un po' i parametri, la perdita dovrebbe sempre cambiare un po'. 
E quindi ci sono un sacco di saluti, perché abbiamo bisogno di guardare un po' la matematica di come funziona e lo faremo nel prossimo paio di lezioni. 
Grazie per le loro grandi domande. 
Ok, quindi la messa a punto è una particolare tecnica di apprendimento di trasferimento in cui il - oh e stai ancora mostrando la tua foto e non le diapositive. 
Quindi la messa a punto è una tecnica di transfer learning dove i pesi (non è proprio la parola giusta che dovremmo dire i parametri) dove i parametri di un modello pre-formato vengono aggiornati con l'allenamento per altre epoche utilizzando un compito diverso da quello usato per il pre-formaggio. 
Quindi il pre-apprendimento del compito potrebbe essere stato la classificazione di ImageNet e quindi il nostro compito diverso potrebbe essere quello di riconoscere i gatti rispetto ai cani. 
Quindi il modo in cui il fastai per default fa la messa a punto è che noi usiamo un'epoca, che, ricordate, è quella di guardare ogni immagine nel set di dati una volta. 
Un'epoca per adattare solo quelle parti del modello necessarie per far funzionare la parte particolare del modello che è specialmente per il vostro set di dati. 
E poi usiamo tutte le epoche che avete chiesto per montare l'intero modello. 
E quindi questo è di più se voi per le persone che potrebbero essere un po' più avanzate vedremo esattamente come funziona più avanti nelle lezioni. 
Allora, perché il trasferimento dell'apprendimento funziona e perché funziona così bene? 
Il modo migliore, secondo me, di guardare la cosa è vedere questo articolo di Zeiler e Fergus, che in realtà sono stati vincitori del 2012 di ImageNet e, curiosamente, le loro intuizioni chiave sono venute dalla loro capacità di visualizzare ciò che succede all'interno di un modello. 
E così la visualizzazione si rivela molto spesso super importante per ottenere grandi risultati. 


Quello che sono stati in grado di fare è stato il loro aspetto - ricordate che vi ho detto che un resnet 34 ha 34 strati? 
Hanno guardato una cosa chiamata AlexNet che era il precedente vincitore del concorso, che aveva solo sette strati. 
All'epoca era considerato enorme e così hanno preso il modello a sette strati e hanno detto: "Com'è il primo strato di parametri? 
E hanno capito come disegnarne un'immagine, giusto? 
E così il primo strato aveva molte e molte caratteristiche, ma qui ce ne sono nove, uno due tre quattro cinque sei sette otto nove. 
Ed ecco come sono fatte nove di quelle immagini. 
Una di esse era qualcosa che poteva riconoscere le linee diagonali dall'alto a sinistra in basso a destra. 
Una di esse era in grado di riconoscere le linee diagonali dal basso a sinistra verso l'alto a destra. 
Una di esse poteva trovare gradienti che andavano dall'alto dell'arancione al basso del blu. 
Alcuni di loro erano in grado, sai, uno di loro era specifico per trovare cose che erano verdi, e così via a destra. 
Quindi, per ognuno di questi nove, sono chiamati filtri o caratteristiche. 
Quindi qualcosa di veramente interessante che hanno fatto è stato guardare ognuno di questi, ognuno di questi filtri, ognuna di queste caratteristiche, e impareremo matematicamente cosa significano realmente nelle prossime lezioni, ma per ora, riconosciamole e diciamo che c'è qualcosa che guarda le linee diagonali e qualcosa che guarda i gradienti e hanno trovato nelle immagini reali in imagenet esempi specifici di parti di foto che corrispondono a quel filtro. 
Quindi per questo filtro in alto a sinistra ci sono nove patch di foto reali che corrispondono a quel filtro e come potete vedere sono tutte linee diagonali. 
E quindi, per quello verde, ecco le parti di foto reali che corrispondono a quello verde. 
Quindi il primo livello è super super semplice e una delle cose interessanti da notare qui è che qualcosa che può riconoscere gradienti e patch di colore e linee è probabilmente utile anche per molti altri compiti, non solo per imagenet. 
Così si può vedere come qualcosa che può fare questo potrebbe essere utile anche per molti altri compiti di computer vision. 
Questo è il livello 2, il livello 2 prende le caratteristiche del livello 1 e le combina. 
quindi non può solo trovare bordi che possono trovare angoli o ripetere modelli di curvatura o semicerchi o cerchi pieni. 
E così si può vedere per esempio ecco un, è un po' difficile visualizzare esattamente questi strati dopo il livello 1. 
Si devono mostrare degli esempi di come sono fatti i filtri. 


Ma qui potete vedere esempi di parti di foto che questi, questo filtro circolare di livello 2 ha attivato su. 
E come potete vedere ha trovato delle cose, con dei cerchi. 
Così interessante questo che è questo tipo di gradiente chiazzato sembra essere molto bravo a trovare i tramonti. 
E questo schema verticale che si ripete è molto bravo a trovare, come le tende e i campi di grano e cose simili. 
Quindi, più ci si allontana, lo strato tre riesce a combinare tutti i tipi di caratteristiche nello strato due. 
E ricordate che qui vediamo solo dodici delle caratteristiche, ma in realtà ce ne sono probabilmente centinaia. 
Non ricordo esattamente in alex net, ma ce ne sono molte. 
Ma quando arriviamo al livello tre, combinando le caratteristiche del livello due, ha già qualcosa che è trovare il testo. 
Quindi questa è una caratteristica che può trovare bit di immagine che contengono testo. 
Ha già qualcosa che può trovare motivi geometrici che si ripetono. 
E si vede che questo non è solo come uno specifico pattern di pixel corrispondente. 
È come un concetto semantico. 
Può trovare cerchi o quadrati che si ripetono o esagoni che si ripetono. 
Fantastico. 
Quindi è davvero come l'informatica, non è solo l'abbinamento di un modello. 
E ricordate che sappiamo che le reti neurali possono risolvere qualsiasi possibile funzione computabile. 
Quindi può certamente farlo. 
Quindi il quarto livello può comunque combinare tutti i filtri del terzo livello in una volta sola. 
E così dal quarto strato abbiamo qualcosa che può trovare volti di cani, per esempio. 
Così si può vedere come ogni strato otteniamo caratteristiche più sofisticate dal punto di vista applicativo. 


Ed è per questo che queste reti neurali profonde possono essere così incredibilmente potenti. 
È anche il motivo per cui il trasferimento dell'apprendimento può funzionare così bene. 
Perché, ad esempio, se volevamo qualcosa che potesse trovare libri. 
E non credo che ci sia una categoria di libri in imagenet. 
Beh, in realtà ha già qualcosa che può trovare il testo come filtro precedente, che immagino debba essere usato per trovare forse una categoria per la biblioteca o qualcosa o uno scaffale. 
Quindi quando si usa il transfer learning si può approfittare di tutte queste caratteristiche pre-apprendimento per trovare cose che sono come combinazioni di queste o caratteristiche esistenti. 
Ecco perché il transfer learning può essere fatto molto più velocemente e con meno dati rispetto agli approcci tradizionali. 
Una cosa importante da capire è che queste tecniche di computer vision non sono solo buone per riconoscere le foto; ci sono tutti i tipi di cose che si possono trasformare in immagini, per esempio questi sono suoni che sono stati trasformati in immagini rappresentando le loro frequenze nel tempo e si scopre che se si converte un suono in questo tipo di immagini si possono ottenere risultati fondamentalmente allo stato dell'arte nel riconoscimento del suono semplicemente usando lo stesso identico resnet learner che abbiamo già visto. 
Dobbiamo sottolineare che è 945, quindi se volete fare una pausa a breve?
Un esempio davvero fico, credo che il nostro primo anno di fastai in esecuzione; uno dei nostri studenti ha creato delle immagini, ha lavorato allo Splunk nell'antifrode, e ha creato immagini di utenti che muovevano il mouse e, se ricordo bene, mentre muovevano il mouse ha praticamente disegnato un'immagine di dove il mouse si muoveva e il colore dipendeva da quanto velocemente si muovevano e queste macchie circolari sono dove hanno cliccato il tasto sinistro o destro del mouse. 
A Splunk quello che ha fatto in realtà per il corso, come progetto per il corso, è stato cercare di vedere se poteva usare queste immagini con esattamente lo stesso approccio che abbiamo visto nella lezione 1 per creare un modello antifrode, e ha funzionato così bene che Splunk ha finito per brevettare un nuovo prodotto basato su questa tecnica e si può effettivamente controllare fuori c'è un post sul blog su internet dove descrivono questo approccio antifrode innovativo che letteralmente è venuto da uno dei nostri studenti davvero sorprendente e brillante e creativo dopo la lezione uno del corso. 


Un altro bell'esempio di questo è guardare diversi virus e trasformarli di nuovo in immagini e si può vedere come hanno ottenuto qui questo è da un giornale, controllare il libro per la citazione, hanno tre esempi di un virus particolare chiamato VB.AT e un altro esempio di un virus particolare chiamato Fakerean e si può vedere in ogni caso le immagini sono tutte un po 'simili ed è per questo che ancora una volta possono ottenere risultati all'avanguardia nel rilevamento dei virus; trasformando le firme del programma in immagini e metterlo attraverso il riconoscimento delle immagini. 
Quindi nel libro troverete una lista di tutti i termini, tutti i termini più importanti, che abbiamo visto finora e che cosa significano. Non ho intenzione di leggerli, ma vi prego, perché questi sono i termini che useremo d'ora in poi e dovete sapere che cosa significano, perché se non lo farete sarete davvero confusi, perché io parlerò di etichette e architetture e modelli e parametri e hanno significati molto specifici e userò quei significati esatti, quindi vi prego di rivedere questo. 
Quindi, per ricordarvi che siamo arrivati a questo punto; abbiamo finito con l'approccio globale di Arthur Samuels e abbiamo sostituito i suoi termini con i nostri termini in modo da avere un'architettura che contiene i parametri come input, i parametri del pozzo e i dati come input in modo che l'architettura più i parametri siano il modello, con gli input che hanno usato per calcolare le previsioni, sono confrontati con le etichette con una funzione di perdita e quella funzione di perdita è usata per aggiornare i parametri molte molte molte volte per renderli migliori e migliori fino a quando la perdita diventa bella e super bassa. 
Quindi questa è la fine del capitolo 1 del libro. 
E' davvero importante guardare il questionario perché il questionario è la cosa in cui si può verificare se si è tolto da questo libro, da questo capitolo il materiale che speriamo si abbia. 
Quindi esaminatelo e tutto ciò di cui non siete sicuri, la risposta è nel testo, quindi tornate a prima nel libro e nel capitolo troverete le risposte. 
C'è anche un'ulteriore sezione di ricerca dopo ogni questionario, per il primo paio di capitoli sono in realtà piuttosto semplici, si spera siano piuttosto divertenti e interessanti; sono cose a cui rispondere alla domanda non basta guardare nel capitolo, bisogna anche andare a pensare e a sperimentare e a cercare su Google e così via. 
Nei capitoli successivi alcune di queste ulteriori ricerche sono progetti piuttosto significativi che potrebbero richiedere qualche giorno o addirittura settimane e quindi dategli un'occhiata, perché si spera che siano un ottimo modo per ampliare la vostra comprensione del materiale. 
Quindi una cosa che Sylvain sottolinea nel libro è che se volete davvero trarre il massimo da questo, dopo ogni capitolo, prendetevi il tempo di sperimentare con il vostro progetto e all'interno dei libri che vi forniamo e poi vedete se potete rifare i quaderni su un nuovo set di dati. 


Forse per il primo capitolo, che potrebbe essere un po' difficile perché non abbiamo mostrato come cambiare le cose, ma per il secondo capitolo, che inizieremo dopo, sarete assolutamente in grado di farlo. 
Ok, allora facciamo una pausa di 5 minuti e torniamo alle 9:55 ora di San Francisco.
Ok, allora bentornati a tutti e penso che abbiamo un paio di domande da fare, quindi Rachel vi prego di portarle via. 
Certo, i filtri sono indipendenti da questo voglio dire che se i filtri sono pre-formati potrebbero diventare meno buoni e rilevare le caratteristiche delle immagini precedenti quando vengono messi a punto? 
Oh questa è un'ottima domanda, quindi, supponendo di aver capito bene la domanda, se si inizia con dire un modello imagenet e poi lo si mette a punto su cani contro gatti per qualche epoca e si ottiene qualcosa che è molto bravo a riconoscere i cani contro i gatti sarà molto meno bravo come modello imagenet dopo di che, quindi non sarà molto bravo a riconoscere gli aerei o i martelli o qualsiasi altra cosa. 
Questo si chiama dimenticanza catastrofica nella letteratura, l'idea che quando si vedono più immagini di cose diverse da quelle che si sono viste prima, si comincia a dimenticare quali sono le cose che si sono viste prima. 
Quindi, se si vuole mettere a punto qualcosa che sia buono in un nuovo compito, ma che continui ad essere buono anche nel compito precedente, bisogna continuare a inserire esempi del compito precedente. 
Quali sono le differenze tra i parametri e gli iper parametri? 
Se sto alimentando un'immagine di un cane come input e poi cambio i parametri iper di dimensione del lotto nel modello, quale sarebbe un esempio di parametro? 
Quindi i parametri sono le cose descritte nella lezione uno che Arthur Samuel ha descritto come le cose che cambiano ciò che fa il modello, ciò che fa l'architettura. 
Quindi iniziamo con questa funzione infinitamente flessibile, la cosa chiamata rete neurale, che può fare qualsiasi cosa e il modo in cui si fa fare una cosa contro l'altra è cambiandone i parametri. 
Sono i numeri che si passano in quella funzione, quindi ci sono due tipi di numeri che si passano nella funzione: ci sono i numeri che rappresentano il vostro input, come i pixel del vostro cane, e ci sono i numeri che rappresentano i loro parametri appresi. 
Quindi, nell'esempio di qualcosa che non è una rete neurale, ma come un programma di gioco a dama come Arthur Samuel avrebbe potuto usare nei primi anni '60 e alla fine degli anni '50, questi parametri potrebbero essere stati cose del tipo: se c'è l'opportunità di prendere un pezzo contro l'opportunità di arrivare alla fine di una tavola, quanto più valore dovrei considerare l'uno contro l'altro. 
Sai che è due volte più importante o è tre volte più importante - che due contro tre - questo sarebbe un esempio di parametro. 
In una rete neurale, i parametri sono un concetto molto più astratto e quindi una comprensione dettagliata di cosa sono verrà nella prossima lezione o due, ma è la stessa idea di base: sono i numeri che cambiano ciò che il modello fa per essere qualcosa che riconosce i tumori maligni, contro i gatti contro i cani contro i colori delle immagini in bianco e nero. 
Mentre l'iperparametro è la scelta dei numeri che si passa alla funzione, alla funzione di adattamento vero e proprio per decidere come avviene il processo di adattamento. 
C'è una domanda: "Sono curioso di sapere il ritmo di questo corso. 
Sono preoccupato che tutto il materiale possa non essere coperto". Dipende da cosa intendi per tutto il materiale. 
Di certo non copriremo tutto nel mondo, quindi sì, copriremo quello che potremo. 
Copriremo quello che possiamo in sette lezioni; di certo non copriremo l'intero libro, se è questo che vi state chiedendo. 


L'intero libro sarà trattato in due o tre corsi. 
In passato sono stati generalmente due corsi per coprire la quantità di materiale del libro, ma vedremo come andrà, perché il libro è piuttosto grande - 500 pagine. 
Quindi quando dice due corsi, intende dire quattordici lezioni? Quattordici, sì, sarebbe come se fossero 14 o 21 lezioni per coprire l'intero libro. 
Anche se, detto questo, alla fine della prima lezione si spera che ci sia abbastanza slancio e che si capisca che leggere il libro in modo indipendente sarà più utile e si avrà anche una specie di comunità di persone sui forum con cui si può uscire e fare domande e così via. 
Quindi nella seconda parte del corso parleremo di come mettere in produzione le cose e quindi, per farlo, dobbiamo capire quali sono le capacità e i limiti dell'apprendimento profondo? Quali sono i tipi di progetti che hanno senso cercare di mettere in produzione? 
E sapete una delle cose chiave che dovrei menzionare nel libro e in questo corso è che nelle prime due o tre lezioni e nei primi capitoli, c'è un sacco di roba che è progettata non solo per i codificatori ma per, per tutti. 
Ci sono molte informazioni su, quali sono le cose pratiche da sapere per far funzionare l'apprendimento profondo. 
E così una di queste cose, le cose che dovete sapere è: "Beh, cos'è l'apprendimento profondo che è veramente buono in questo momento? 
Quindi riassumerò quello che il libro dice di questo, ma ci sono il tipo di quattro aree chiave che abbiamo come applicazioni nel Fastai: la visione computerizzata, il testo, la tabulazione, e quello che ho chiamato qui "Recsys", per i sistemi di raccomandazione e nello specifico una tecnica chiamata filtraggio collaborativo che abbiamo brevemente visto... . 
Scusate un'altra domanda, ci sono altri pesi preformati disponibili oltre a quelli di Imagenet che possiamo usare? 
Se sì, quando dovremmo usarne altri e quando Imagenet? 
Oh, questa è davvero una bella domanda. 
Quindi sì, ci sono molti modelli pre-formati, e un modo per trovarli... 
E inoltre al momento ci state mostrando.. 
Ok grande. 
Un ottimo modo per trovarli è quello di cercare i modelli dello zoo, che è un nome comune per i luoghi che hanno un sacco di modelli diversi. 
E quindi ecco un sacco di modelli di zoo. 
Oppure potete cercare modelli pre-formati. 
E quindi sì, ce n'è un bel po', purtroppo non così vasta come vorrei che la maggior parte è ancora su Imagenet o simili tipi di foto generali. 
Per esempio l'imaging medico non c'è quasi 


Ci sono molte opportunità per le persone di creare modelli pre-formati specifici per il dominio, ma si tratta di un'area ancora poco sfruttata perché non ci sono abbastanza persone che lavorano sul transfer learning. 
Ok, come stavo dicendo, abbiamo un po' di queste quattro applicazioni di cui abbiamo parlato e l'apprendimento profondo è abbastanza, sai, abbastanza buono in tutti quei dati tabulari come fogli di calcolo e tabelle di database è un'area dove l'apprendimento profondo non è sempre la scelta migliore, ma è particolarmente buono per le cose che coinvolgono variabili ad alta cardinalità, cioè variabili che hanno molti e molti livelli discreti come il codice postale o l'ID del prodotto o qualcosa del genere. 
L'apprendimento profondo è davvero ottimo per chi è particolarmente interessato. 
Per quanto riguarda il testo, è ottimo per cose come la classificazione e la traduzione. 
In realtà è terribile per la conversazione e questo è stato un aspetto che è stato una grande delusione per molte aziende. Ho cercato di creare queste cose come i bot per la conversazione, ma in realtà l'apprendimento profondo non è buono a fornire informazioni accurate, è buono a fornire cose che suonano accurate e convincenti, ma che non abbiamo ancora grandi modi per essere sicuri che siano corrette. 
Un grande problema per i sistemi di filtraggio collaborativo dei sistemi di raccomandazione è che l'apprendimento profondo si concentra sul fare previsioni che non significano necessariamente creare raccomandazioni utili. 
Vedremo cosa significa tra un attimo. 
L'apprendimento profondo è buono anche nel multimodale, il che significa cose in cui hai diversi tipi di dati, quindi potresti avere alcuni dati tabellari tra cui una colonna di testo e un'immagine, poi alcuni dati di filtraggio collaborativo e combinarli tutti insieme è qualcosa in cui l'apprendimento profondo è davvero bravo. 
Quindi, per esempio, mettere le didascalie sulle foto è qualcosa in cui l'apprendimento profondo è piuttosto bravo, anche se, ancora una volta, non è molto bravo ad essere preciso. 
Quindi quello che si sa potrebbe dire è una foto di due uccelli quando in realtà è una foto di tre uccelli e poi in quest'altra categoria ci sono tante e tante cose che si possono fare con l'apprendimento profondo essendo creativi sull'uso di questo tipo di altri approcci basati su applicazioni, per esempio un approccio che abbiamo sviluppato per l'elaborazione del linguaggio naturale chiamato ULMFit che impareremo durante il corso. 
Si scopre che è anche fantastico che tu stia facendo l'analisi delle proteine. 
Se si pensa alle diverse proteine come a parole diverse e sono in una sequenza che ha un qualche tipo di stato e significato, si scopre che ULMFit funziona molto bene per l'analisi delle proteine. 
Così spesso si tratta di essere un po' creativi. 
Quindi, per decidere come per il prodotto che si sta cercando di costruire è un apprendimento profondo che funziona bene per esso, alla fine devi solo provare e vedere, ma se fai una ricerca sai che se fai una ricerca si spera di poter trovare esempi sulle persone che hanno provato qualcosa di simile anche se non ci riesci, questo non significa che non funzionerà. 
Così, per esempio, ho menzionato il problema del filtraggio collaborativo, dove una raccomandazione e una previsione non sono necessariamente la stessa cosa. 
Su Amazon, per esempio, lo si vede spesso. 
Così ho comprato un libro di Terry Pratchett e poi Amazon ha cercato per mesi di farmi comprare altri libri di Terry Pratchett. 
Ora questo deve essere perché il loro modello predittivo diceva che le persone che hanno comprato un particolare libro di Terry Pratchett probabilmente compreranno anche altri libri di Terry Pratchett. 
Ma dal punto di vista del come bene questo cambierà il mio comportamento d'acquisto: probabilmente no, giusto, come se mi piacesse quel libro so già che mi piace quell'autore e so già che come probabilmente hanno scritto altre cose quindi andrò a comprarlo comunque. 
Quindi questo sarebbe un esempio di come Amazon probabilmente non è molto intelligente, quassù mi stanno in realtà mostrando previsioni di filtraggio collaborativo piuttosto che capire come ottimizzare una raccomandazione. 


Quindi una raccomandazione ottimizzata sarebbe qualcosa di più simile a quello che potrebbe fare il vostro libraio umano locale, dove potrebbe dire: "Oh! Ti piace Terry Pratchett, beh lascia che ti parli di altri tipi di scrittori fantascientifici di fantascienza comica sulla stessa vena di cui potresti non aver mai sentito parlare prima". 
Quindi la differenza tra le raccomandazioni e le previsioni è super importante. 
Quindi volevo parlare di una questione molto importante che riguarda l'interpretazione dei modelli e per un caso di studio ho pensato di scegliere qualcosa di super importante in questo momento, che è un modello in questo articolo. 
Una delle cose che cercheremo di fare in questo corso è imparare a leggere i giornali. 
Quindi ecco un saggio che mi piacerebbe che tutti leggessero e che si chiama "Alta temperatura e alta umidità riducono la trasmissione di COVID-19". 
Ora, questa è una questione molto importante perché se l'affermazione di questo documento è vera, allora significa che si tratterà di una malattia stagionale e se questa è una malattia stagionale e avrà enormi implicazioni politiche. 
Quindi cerchiamo di scoprire come è stato modellato e di capire come interpretare questo modello. 
Questa è un'immagine chiave del giornale e quello che hanno fatto qui è che hanno preso un centinaio di città in Cina e hanno tracciato la temperatura su un asse, in gradi Celsius, e R sull'altro asse, dove R è una misura di trasmissibilità. 
Dice per ogni persona che ha questa malattia quante persone infetteranno in media. 
Quindi se R è inferiore a 1, la malattia non si diffonderà. 
Se R è superiore a 2, si diffonderà incredibilmente rapidamente. 
Fondamentalmente R, sapete, qualsiasi R elevato creerà un impatto di trasmissione esponenziale. 
E si può vedere che in questo caso hanno tracciato una linea che si adatta al meglio qui. 
Poi hanno affermato che c'è una relazione particolare in termini di formula che R è 1,99 meno 0,023 volte la temperatura. 
Quindi la preoccupazione più ovvia che avrei avuto guardando questa immagine è che questo potrebbe essere solo casuale, forse non c'è affatto una relazione, ma solo se si sceglie un centinaio di città a caso, forse a volte mostrano questo livello di relazione. 
Quindi un modo semplice per vedere la cosa sarebbe quello di farlo in un foglio di calcolo. 
Ecco un foglio di calcolo. 
Quello che ho fatto è stato una specie di occhio su questi dati e ho indovinato quali sono i gradi centigradi medi. 
Penso che siano circa 5. 
E la deviazione standard di gradi centigradi. 


Credo che probabilmente siano anche circa 5. 
E poi ho fatto la stessa cosa per R. 
Penso che la R cattiva mi sembri circa 1,9. 
E sembra che la deviazione standard di R sia probabilmente circa 0,5. 
Quindi quello che ho fatto è stato saltare qui e ho creato un valore normale casuale, quindi un valore casuale da una distribuzione normale, quindi una curva a campana, con quella particolare media e deviazione standard della temperatura e quella particolare media e deviazione standard di R. 
E così questo sarebbe un esempio di una città che potrebbe essere in questo insieme di dati di un centinaio di città. 
Qualcosa con 9 gradi Celsius e R di 1,1; quindi sarebbe 9 gradi Celsius e R di 1,1, qualcosa di simile qui. 
Quindi ho appena copiato questa formula 100 volte. 
Quindi qui ci sono un centinaio di città che potrebbero essere in Cina a destra, dove questo è supporre che non c'è relazione tra temperatura e R a destra. 
Sono solo numeri casuali e così ogni volta che ricalcolo che così se ho colpito il controllo è uguale al controllo, sarà solo ricalcolare a destra. 
Ottengo numeri diversi, perché sono casuali. 
E così potete vedere in alto qui ho poi la media di tutte le temperature e la media di tutte le R e la media di tutte le temperature varia e varia anche la media di tutte le R. 
Quindi quello che ho fatto è stato copiare qui quei numeri casuali. 
Facciamolo davvero. 
Quindi vado a copiare questi 100 numeri casuali e li incollo qui, qui, qui. 
E così ora ho 1 2 3 3 4 4 5 5 6 ho 6 tipi di gruppi di 100 città. 
Va bene, e allora fermiamo quelli che non cambiano più a caso, fissandoli semplicemente lì in pietra.
Ok, ora che le ho incollate, ho 6 esempi di come potrebbero essere un centinaio di città se non ci fosse alcuna relazione tra temperatura e R. 
Ho la loro temperatura media e R in ognuno di questi sei esempi. 
Quello che ho fatto, potete vedere qui, almeno per il primo, è che l'ho tracciato, giusto?  


Si può vedere, in questo caso, che in realtà c'è una leggera pendenza positiva. 
In realtà ho calcolato la pendenza per ciascuno di essi, semplicemente utilizzando la funzione di pendenza in Microsoft Excel. 
Si può vedere che in realtà, in questo caso particolare, è solo casuale - cinque volte è stato negativo, ed è ancora più negativo del loro 0,023. 
Quindi si può dire che, in un certo senso, corrisponde alla nostra intuizione, ovvero che la pendenza della linea che abbiamo qui, è qualcosa che spesso può accadere assolutamente in modo del tutto casuale. 
Non sembra indicare alcun tipo di relazione reale. 
Se volevamo che quella pendenza fosse più sicura, avremmo dovuto guardare più città. 
Qui ho 3.000 numeri generati casualmente. 
Qui si vede che la pendenza è 0,00002, giusto? 
È quasi esattamente zero, che è quello che ci aspetteremmo, quando in realtà non c'è nessuna relazione tra C e R, e in questo caso non c'è - sono tutte casuali. 
Se poi guardiamo tante e tante città generate in modo casuale, allora possiamo dire: oh sì, non c'è pendenza. 
Ma se ne guardi solo un centinaio, come abbiamo fatto qui, vedrai le relazioni in modo del tutto casuale, molto, molto spesso. 
Quindi è qualcosa che dobbiamo essere in grado di misurare. 
Un modo per misurare questo è usare una cosa chiamata p-valore. 
Un p-valore, ecco come funziona un p-valore: partiamo da qualcosa chiamato ipotesi nulla. 
L'ipotesi nulla è fondamentalmente il nostro punto di partenza. 
Il nostro punto di partenza potrebbe essere, oh non c'è relazione tra temperatura e R. 
E poi raccogliamo alcuni dati e (Rachel: hai spiegato cos'è R?) Sì, l'ho fatto. 
R è la trasmissibilità del virus. 
Quindi raccogliamo dati di variabili indipendenti e dipendenti - in questo caso la variabile indipendente è la cosa che pensiamo possa causare la variabile dipendente. 
Qui la variabile indipendente sarebbe la temperatura, la variabile dipendente sarebbe R. 


Quindi qui abbiamo raccolto i dati - ci sono i dati che sono stati raccolti in questo esempio, e poi diciamo in quale percentuale del tempo vedremmo questa quantità di relazione, che è una pendenza di 0,023 per caso? 
E come abbiamo visto, un modo per farlo è quello che chiameremmo una simulazione, che è generare numeri casuali - un insieme di 100 coppie di numeri casuali, un mucchio di volte, e vedere quanto spesso si vede questa relazione. 
In realtà, però, non dobbiamo farlo per forza. 
In realtà c'è una semplice equazione che possiamo usare per saltare direttamente a questo numero, che è, quale percentuale del tempo vedremmo per caso questa relazione? 
E questo è fondamentalmente quello che sembra. 
Abbiamo l'osservazione più probabile, che in questo caso sarebbe se non ci fosse una relazione tra le temperature. 
Allora la pendenza più probabile sarebbe zero, e a volte si ottengono pendenze positive per caso, e a volte si ottengono pendenze piuttosto piccole, e a volte si ottengono grandi pendenze negative per caso. 
E così, più grande è il numero, meno probabilità ci sono che si verifichi, sia che si tratti del lato positivo che di quello negativo. 
Nel nostro caso, la nostra domanda era: quanto spesso otterremo meno di 0,023 negativo? 
In realtà sarebbe da qualche parte qui sotto. 
In realtà ho copiato questo da Wikipedia, dove cercavano numeri positivi, e così hanno colorato in quest'area sopra il numero. 
Questo è il valore p, e non ci interessa la matematica, ma c'è una semplice piccola equazione che si può usare per capire direttamente questo numero - il valore p - dai dati. 
Questo è il modo in cui quasi tutti i risultati della ricerca medica tendono ad essere mostrati, e la gente si concentra davvero su questa idea dei valori p. 
E infatti, in questo particolare studio, come vedremo tra un momento, hanno riportato i valori p. 
Probabilmente molti di voi hanno visto i valori p nelle vostre vite precedenti. 
Vengono fuori in molti settori diversi. 
Il fatto è questo: sono terribili. 
Quasi sempre non dovreste usarli. 
Non fidatevi solo di me. 
Fidatevi dell'Associazione Statistica Americana. 


Essi sottolineano sei cose sui valori p, e questi includono: i valori p non misurano la probabilità che l'ipotesi sia vera, o, la probabilità che i dati siano stati prodotti solo per scelta casuale. 
Ora lo sappiamo perché abbiamo appena visto che, se usiamo più dati, se campiamo tremila città a caso invece di cento, otteniamo un valore molto più piccolo. 
Quindi i valori p non ti dicono solo quanto è grande una relazione, ma ti dicono anche una combinazione di questo e quanti dati hai raccolto. 
Quindi non misurano la probabilità che l'ipotesi sia vera. 
Quindi, le conclusioni e le decisioni politiche non dovrebbero basarsi sul fatto che un valore p superi una certa soglia. 
Il valore p non misura l'importanza di un risultato, perché, anche in questo caso, potrebbe semplicemente dirvi che avete raccolto molti dati, il che non vi dice che i risultati siano effettivamente di una qualche importanza pratica. 
Di per sé, non fornisce una buona misura di prova. 
Frank Harrell, che è una persona che ho letto nel suo libro, ed è una parte molto importante del mio apprendimento. 
È un professore di biostatistica, ha una serie di ottimi articoli su questo argomento. 
Dice che i test delle ipotesi nulle e i valori p hanno fatto un danno significativo alla scienza. 
Ha scritto un altro pezzo chiamato "i test di significato delle ipotesi nulle non hanno mai funzionato". 
Vi ho mostrato cosa sono i valori p in modo che sappiate perché non funzionano, non per poterli usare. 
Ma sono una parte super importante dell'apprendimento delle macchine, perché vengono fuori di continuo. 
Quando la gente dice, questo è il modo in cui decidiamo se il tuo farmaco ha funzionato, o se c'è una relazione epidemiologica, o qualsiasi altra cosa. 
E in effetti, i valori p appaiono in questo documento. 
Nel documento, essi mostrano i risultati di una regressione lineare multipla. 
Mettono tre stelle accanto a qualsiasi relazione che abbia un valore p di 0,01 o meno. 
C'è qualcosa di utile da dire su un piccolo valore p, come 0,01 o meno. 
Qual è la cosa che stiamo guardando non è successo, probabilmente non è successo per caso, giusto? 
Il più grande errore statistico che le persone fanno sempre è che vedono che un valore p non è inferiore a 0,05 e poi giungono alla conclusione errata che non esiste alcuna relazione, giusto? 


Il che non ha alcun senso perché, come diciamo che hai avuto solo tre punti dati, allora quasi certamente non avrai abbastanza dati per avere un valore p inferiore a 0,05 per qualsiasi ipotesi. 
Quindi, come il modo di verificare, è di tornare indietro e dire: e se scegliessi l'ipotesi nulla esattamente opposta? 
E se la mia ipotesi nulla fosse che c'è una relazione tra temperatura e R? 
Allora ho abbastanza dati per rifiutare quell'ipotesi nulla, va bene? E se la risposta è no, allora non hai abbastanza dati per trarre alcuna conclusione, va bene? 
Quindi in questo caso hanno abbastanza dati per essere sicuri che ci sia una relazione tra temperatura e R. 
Ora, questo è strano perché abbiamo appena guardato il grafico, e abbiamo fatto un po' il retro di un po' di back-of-the-envelope in Excel e abbiamo pensato che questo è, potrebbe, potrebbe essere casuale. 
Quindi ecco dove sta il problema. 
Il grafico mostra quella che chiamiamo una relazione univariata. 
Una relazione univariata mostra la relazione tra una variabile indipendente e una variabile dipendente, ed è quello che normalmente si può mostrare in un grafico. 
Ma in questo caso hanno fatto un modello multivariato in cui hanno guardato alla temperatura, e all'umidità, e al PIL pro capite, e alla densità della popolazione, e quando si mettono tutte queste cose nel modello allora si finisce per ottenere risultati statisticamente significativi per la temperatura e l'umidità. 
Perché questo accade? 
Beh, il motivo è che tutte queste variazioni nei punti blu non sono casuali. 
C'è un motivo per cui sono diverse, giusto? 
E le ragioni sono: le città più dense avranno una trasmissione maggiore, per esempio, e probabilmente più umide avranno una trasmissione minore. 
Quindi, quando si fa un modello multivariato, in realtà permette di essere più sicuri dei risultati, giusto? 
Ma il valore p, come rilevato dall'American Statistical Association, non ci dice se questo sia di importanza pratica. 
La cosa che ci dice se questo è di importanza pratica come importanza, è la pendenza effettiva che si trova. 
E così in questo caso l'equazione che ne risulta è che R = tre punti nove sei otto meno tre punti O punto O tre otto per temperatura meno punto zero due quattro per umidità relativa questa è l'equazione che è praticamente importante. 
Bene, possiamo di nuovo fare un piccolo retro della busta qui, semplicemente mettendolo in Excel. 
Diciamo che c'era un posto in cui aveva una temperatura di dieci gradi centigradi e un'umidità di quaranta, allora se questa equazione è corretta R sarebbe circa due punti sette da qualche parte con la temperatura di 35 gradi centigradi e un'umidità di ottanta sarà circa il punto otto. 


Quindi questo è praticamente importante? Oh mio Dio sì, vero? 
Due città diverse, con climi diversi possono esserlo, se sono uguali in ogni altro modo, e questo modello è corretto allora una città non avrebbe nessuna diffusione di malattie (perché R è meno di 1), una avrebbe una massiccia esplosione esponenziale. 
Quindi possiamo vedere da questo modello che se la modellazione è corretta, allora questo è un risultato molto significativo. 
Quindi questo è il modo in cui si determina il significato pratico dei vostri modelli non con valori p, ma con l'osservazione di una sorta di risultati reali. 
Quindi come si pensa all'importanza pratica di un modello e come si trasforma un modello predittivo in qualcosa di utile nella produzione.
Così ho passato molti anni a pensare a questo, e ho effettivamente creato un documento su di esso insieme ad altri grandi personaggi.
"Progettare grandi prodotti di dati" 
E questo si basa in gran parte su dieci anni di lavoro che ho svolto in un'azienda che ho fondato e che si chiama Optimal Decisions Group. 
E Optimal Decisions Group si è concentrato sulla questione di aiutare le compagnie di assicurazione a capire quali prezzi fissare.
E le compagnie di assicurazione fino a quel momento si erano concentrate sulla modellazione predittiva.
Gli attuari, in particolare, hanno passato il loro tempo a cercare di capire quanto è probabile che si verifichi un incidente e, se lo si fa, quanto danno si potrebbe avere e, in base a ciò, cercare di capire quale prezzo fissare per la polizza. 
Quindi, per questa azienda, quello che abbiamo fatto è stato decidere di usare un approccio diverso che ho finito per chiamare l'approccio della trasmissione, che è descritto qui per fissare i prezzi dell'assicurazione e per fare tutti gli altri tipi di cose.
E così, per l'esempio dell'assicurazione, l'obiettivo per una compagnia di assicurazioni sarebbe come massimizzare il mio profitto, diciamo, di cinque anni.
E poi, quali sono gli input che possiamo controllare, quali sono le leve che io chiamo leve - quindi in questo caso sarebbe il prezzo che posso fissare.
E poi i dati sono dati che possono dirvi come cambiate le vostre leve e come questo cambia il vostro obiettivo.
Quindi, se comincio ad aumentare il mio prezzo per le persone che probabilmente si schianteranno con la loro auto, allora ne otterremo meno, il che significa che avremo meno costi, ma allo stesso tempo avremo anche meno entrate, per esempio.
Quindi, per collegare lì una sorta di leve all'obiettivo attraverso i dati che raccogliamo, costruiamo modelli che descrivono come le leve influenzano l'obiettivo.
E tutto questo è come se sembrasse abbastanza ovvio quando lo dici così, ma quando abbiamo iniziato a lavorare con Optimal Decisions nel 1999, nessuno lo faceva nel settore assicurativo,
Tutti nel settore assicurativo stavano semplicemente facendo un modello predittivo per indovinare la probabilità che le persone si schiantassero con la propria auto, e poi il prezzo è stato fissato aggiungendo il 20% o altro.
È stato fatto in un modo molto ingenuo.


Quindi quello che ho fatto è che, sapete, nel corso di molti anni ho preso questo processo di base e ho cercato di aiutare molte aziende a capire come utilizzarlo per trasformare i modelli predittivi in azioni.
Quindi il punto di partenza per ottenere effettivamente valore in un particolare modello è pensare a quello che si sta cercando di fare, e si sa quali sono le fonti di valore in quella cosa che si sta cercando di fare.
Le leve - quali sono le cose che si possono cambiare? 
Ad esempio, a cosa serve un modello predittivo se non si può fare niente, giusto? 
Trovare il modo di trovare quali dati non si hanno, quali sono adatti, quali sono disponibili, e poi pensare a quali approcci all'analisi si possono adottare.
E poi super importante, come bene, puoi effettivamente implementare, sai, quei cambiamenti. 
E super super importante, come si fa a cambiare le cose quando l'ambiente cambia. 
E, sapete, è interessante notare che molte di queste cose sono aree in cui non c'è molta ricerca accademica. 
Ce n'è un po'. 
E alcuni dei lavori che hanno riguardato in particolare la "manutenzione" di simili; Come decidete quando il vostro modello di machine learning va ancora bene? 
Come si fa ad aggiornarlo nel tempo? 
Hanno avuto come molti molti molti moltissime citazioni, ma non compaiono molto spesso perché molta gente è così concentrata sulla matematica. 
Sapete. 
E poi c'è tutta la questione del tipo "Quali sono i vincoli che ci sono in tutta questa faccenda? Quindi, quello che troverete nel libro, è che c'è un'intera appendice che passa in realtà attraverso ognuna di queste sei cose. 
E ha un intero elenco di esempi. 
Quindi questo è un esempio di come ci piace pensare al valore. 
E un sacco di domande che le aziende e le organizzazioni possono usare per provare a pensare, sai, a tutti questi diversi pezzi del vero e proprio puzzle di mettere le cose in produzione e di fatto in un prodotto efficace. 
Abbiamo una domanda. 
Certo, solo un momento. 
Stavo per dire, quindi date un'occhiata a questa appendice perché in realtà è apparsa originariamente come un post del blog e penso che, a parte i miei covid-19 post che ho fatto con Rachel, sia in realtà il post più popolare che abbia mai scritto. 


Ha avuto centinaia di migliaia di visualizzazioni. 
E rappresenta una sorta di vent'anni di intuizioni conquistate duramente, come ad esempio il modo in cui si ottiene valore dall'apprendimento automatico e dalla pratica e cosa si deve chiedere. 
Quindi, per favore, date un'occhiata, perché speriamo che lo troverete utile. 
Quindi, quando pensiamo a "come", pensiamo a questo per la questione di come la gente dovrebbe pensare al rapporto tra stagionalità e trasmissibilità del covid-19, è necessario scavare molto profondamente nelle domande su come oh non solo su cosa sono quei numeri nei dati, ma su cosa sembra davvero giusto. 
Quindi una delle cose che mostrano nel giornale è la mappa vera e propria, il diritto di temperatura e umidità e il diritto di R. 
E si può vedere come, non sorprendentemente, che l'umidità e la temperatura in Cina sono ciò che chiameremmo auto-correlate. 
Vale a dire che i luoghi che sono vicini l'uno all'altro, in questo caso geograficamente, hanno temperature e umidità simili. 
E quindi, come questo, in realtà mette molto in discussione i valori di p che hanno ragione. 
Perché non si può pensare che queste siano un centinaio di città completamente separate. 
Perché quelle che sono vicine tra loro hanno probabilmente un comportamento molto simile, quindi forse dovresti pensare a loro come a un piccolo numero di città, sai, di una sorta di geografie più grandi. 
Quindi questi sono i tipi di cose che quando si guarda in realtà in un modello che ti piace, devi pensare a quali sono, quali sono i limiti?  
Ma poi, per decidere bene, cosa significa? 
Che cosa devo fare in proposito? 
Devi pensarla da questo tipo di utilità, da questo tipo di fine a fine, quali sono le azioni che posso intraprendere? 
Quali sono l'ordine e il punto di vista dei risultati?  
Non solo la verifica di ipotesi nulle. 
Quindi in questo caso, per esempio, ci sono fondamentalmente quattro possibili modi chiave in cui questo potrebbe finire. 
Potrebbe finire che ci sia davvero una relazione tra temperatura e R, o almeno così è, ma la parte destra lo è. 
Oppure non c'è una vera relazione tra temperatura e R. 
E potremmo agire partendo dal presupposto che ci sia una relazione. 


Oppure potremmo agire partendo dal presupposto che non ci sia una relazione. 
E così lei vuole guardare a ciascuna di queste quattro possibilità e dire, come si deve, quali sarebbero le conseguenze economiche e sociali?  
E sapete che ci sarà un'enorme differenza nelle vite perse e sapete che le economie sono in crisi e quant'altro - sapete per ognuna di queste quattro. 
Il giornale ha mostrato, se il loro modello è corretto, qual è il probabile valore di R a marzo per ogni città del mondo. 
E il probabile valore di R a luglio per ogni città del mondo. 
E così, per esempio, se si guarda ad una sorta di New England e New York, la previsione qui è e anche Ovest, l'altra la costa stessa della costa occidentale è che a luglio la malattia smetterà di diffondersi. 
Ora sai che se questo accadrà, se hanno ragione allora, sarà un disastro perché penso che è molto probabile che in America e anche nel Regno Unito, la gente dirà "Oh, a quanto pare questa malattia non è un problema, lo sai. 
Non è decollata per niente. 
Gli scienziati si sbagliavano". Le persone torneranno alla loro vita quotidiana precedente e potremo vedere cosa è successo nel 1918 con un virus influenzale come il secondo virus dell'influenza. 
Quando l'inverno potrebbe essere molto peggiore dell'inizio. 
Quindi, come se ci fossero questi enormi potenziali impatti politici a seconda che questo sia vero o falso. 
E così, a pensarci bene. 
Sì? Volevo anche dire che sarebbe molto irresponsabile pensare "oh l'estate risolverà tutto". 
Non dobbiamo agire adesso".  Solo che si tratta di qualcosa che cresce in modo esponenziale e che potrebbe fare un danno enorme. 
Sì, sì, va bene. 
A proposito, l'ha già fatto. 
Se date per scontato che ci sarà una stagionalità e che l'estate sistemerà le cose, allora potrebbe portarvi ad essere apatici adesso. 
Se si suppone che non ci sia stagionalità e poi c'è, allora si potrebbe finire per creare un livello più grande di aspettativa di distruzione che in realtà accade e finire con la vostra popolazione che è ancora più apatica che conoscete, in modo da essere apatici. 
Sbagliare in qualsiasi direzione potrebbe essere un problema. 
Quindi uno dei modi in cui tendiamo ad affrontare questo problema, con questo tipo di modellazione, è cercare di pensare ai precedenti. 


Quindi i nostri precedenti sono fondamentalmente cose in cui noi, sapete, invece di avere solo un'ipotesi nulla, cerchiamo di iniziare con un'ipotesi che ci piaccia, cosa è più probabile?  
Giusto, quindi in questo caso se la memoria serve bene penso che sappiamo che come i virus dell'influenza diventano inattivi a 27 gradi centigradi sappiamo che come il freddo, i coronavirus freddi sono stagionali. 
La pandemia influenzale del 1918 era stagionale. 
In ogni paese e città che è stata studiata finora, ci sono stati diversi studi come questo. 
Finora hanno sempre trovato relazioni climatiche. 
Quindi, forse, diremmo: "La credenza precedente è che questa cosa sia probabilmente stagionale". E allora diremmo: "Beh, questo particolare documento aggiunge delle prove". Ciò dimostra quanto sia incredibilmente complesso utilizzare un modello nella pratica per le discussioni sulle politiche, ma anche per le decisioni organizzative. 
Perché, sapete, ci sono sempre delle complessità, ci sono sempre delle incertezze. 
E quindi bisogna pensare alle utilità, sapete. 
E le vostre migliori ipotesi e cercare di combinare tutto insieme nel miglior modo possibile. 
Ok... 
Quindi, detto questo. 
È comunque bello poter mettere in funzione i nostri modelli anche se, sapete, anche solo un modello predittivo a volte è utile da solo. 
A volte è utile per prototipare qualcosa, a volte deve far parte di un quadro più ampio. 
Quindi, piuttosto che cercare di creare un enorme modello end-to-end. 
Abbiamo pensato di mostrarvi come far funzionare il vostro modello Pytorch FastAI. 
Nel modo più grezzo possibile. 
In modo che da lì si possa costruire sopra, come si vuole. 
Quindi, per farlo, scaricheremo e cureremo il nostro dataset. 
E voi farete la stessa cosa. 
Allenerai il tuo modello, su quel dataset, e poi creerai un'applicazione, e poi la ospiterai. 


Giusto? Ora, ci sono molti modi per curare un set di dati di immagini; potresti avere alcune foto sul tuo computer, ci potrebbero essere cose al lavoro che puoi usare. 
Uno dei più semplici, però, è quello di scaricare materiale da internet. 
Ci sono molti servizi per scaricare roba da internet. 
Qui useremo la ricerca di immagini Bing. 
Perché sono facilissimi da usare. 
Molti altri tipi di cose facili da usare richiedono la violazione dei Termini di servizio dei siti web. 
Quindi non vi mostreremo come farlo. 
Ma ci sono molti esempi che vi mostrano come farlo. 
Quindi, se volete, potete controllarli anche voi. 
La ricerca di immagini Bing è in realtà abbastanza grande, almeno per il momento. 
Queste cose cambiano molto, quindi tieni d'occhio il nostro sito web per vedere se abbiamo cambiato la nostra raccomandazione. 
Il problema maggiore con Bing Image Search è che il processo di registrazione è un incubo, almeno al momento. 
Una delle parti più difficili di questo libro è proprio l'iscrizione alla loro maledetta API. 
Il che richiede di passare attraverso Azure. 
Si chiama Servizi Cognitivi - Servizi Cognitivi Azzurri. 
Quindi faremo in modo che tutte queste informazioni siano presenti sul sito web per permettervi di seguire le modalità di iscrizione. 
Quindi partiremo dal presupposto che tu ti sia già iscritto. 
Ma potete trovarlo, andate: Bing, Bing Image Search API. 
E al momento ti danno sette giorni con una quota piuttosto alta gratis. 
E poi, dopo di che, puoi continuare ad usarla quanto vuoi, ma in un certo senso la limitano a tre transazioni al secondo o qualcosa del genere. 


Che è ancora molto. 
Puoi ancora farne migliaia gratis, per cui al momento è abbastanza grande anche gratis. 
Quindi quello che succederà è che quando ti iscriverai a Bing Image Search, o uno qualsiasi di questi servizi, ti daranno una chiave API. 
Quindi basta sostituire il 'XXX' qui con la chiave API che ti danno. 
Ok, quindi questa si chiamerà "chiave". 
In effetti, facciamolo qui. 
Ok, quindi metterai la tua chiave e poi c'è una funzione che abbiamo creato chiamata search_images_bing che è solo una piccolissima funzione. 
Come potete vedere, sono solo due righe di codice -- Stavo solo cercando di risparmiare un po' di tempo, il che richiederà un po' di tempo, che prenderà la vostra chiave API e qualche termine di ricerca e restituirà una lista di URL che corrispondono a quel termine di ricerca. 
Come potete vedere per usare questo particolare servizio dovete installare un particolare pacchetto, quindi vi mostriamo come fare anche questo sul sito. 
Così, una volta fatto questo, sarete in grado di eseguire questo e questo vi restituirà per impostazione predefinita 150 URL. 
Ok, quindi fast.ai è dotato di una funzione download_url, quindi scarichiamo una di queste immagini solo per controllare e aprirla. 
E così quello che ho fatto è stato cercare "orso grizzly" e qui ho un orso grizzly. 
Quindi quello che ho fatto è stato dire, ok, proviamo a creare un modello in grado di riconoscere gli orsi grizzly contro gli orsi neri contro gli orsacchiotti, in modo che io possa scoprirlo. 
Potrei creare un sistema di riconoscimento video vicino al nostro campeggio, quando siamo in campeggio, che mi dia degli avvertimenti sugli orsi, ma se è un orsacchiotto che viene, non mi avverte e mi sveglia, perché non sarebbe per niente spaventoso. 
Così poi mi limito a esaminare ognuno di questi tre tipi di orso, creare una directory con il nome di grizzly o nero o orsacchiotto cercato Bing per quel particolare termine di ricerca insieme all'orso e scaricare. 
E così anche il download_images è una funzione fast.ai. 
Dopodiché posso chiamare get_image_files che è una funzione fast.ai che restituirà ricorsivamente tutti i file di immagine all'interno di questo percorso. 
E si può vedere che mi ha dato orsi/nero e poi un sacco di numeri. 
Quindi una delle cose a cui bisogna stare attenti è che molte delle cose che si scaricano si riveleranno come non immagini e si romperanno. 
Quindi potete chiamare verify_images per controllare che tutti questi nomi di file siano immagini reali. 


E in questo caso non ne ho avuti di falliti, quindi è vuoto. 
Ma se ne avevi un po', allora chiamavi Path.unlink per scollegarti. 
Path.unlink fa parte della libreria standard Python e cancella un file. 
E map è qualcosa che chiamerà questa funzione per ogni elemento di questa collezione. 
Questo fa parte di una speciale classe fast.ai chiamata "L". 
Fondamentalmente è una specie di mix tra la classe della lista delle librerie standard Python e una classe array numpy.
Poi ne impareremo di più in seguito in questo corso, ma fondamentalmente cerca di rendere super facile fare una programmazione in stile più funzionale in Python. 
Quindi in questo caso sta per scollegare tutto ciò che è nella lista dei fallimenti, che è probabilmente quello che vogliamo ora, perché ci sono tutte le immagini che non riescono a verificare. 
Va bene, quindi ora abbiamo un percorso che contiene un intero gruppo di immagini e sono classificate in base al nero, grizzly, o teddy, in base alla cartella in cui si trovano. 
e quindi per creare così creeremo un modello. 
e quindi per creare un modello la prima cosa che dobbiamo fare è dire a fast.ai che tipo di dati abbiamo e come è strutturato. 
Ora in parte nella Lezione 1 del corso abbiamo fatto questo usando quello che chiamiamo un metodo di fabbrica, cioè abbiamo detto che image_data_loader parte dal nome, e questo ha fatto tutto per noi. 
Questi metodi factory vanno bene per i principianti, ma ora ci occupiamo della Lezione 2. 
Non siamo più dei principianti, quindi vi mostreremo il modo super super flessibile di usare i dati in qualsiasi formato vogliate, e si chiama DataBlock API. 
E così l'API DataBlock si presenta in questo modo. 
Ecco le API DataBlock. 
Tu dici a fast.ai cos'è la tua variabile indipendente e cos'è la tua variabile dipendente. 
Quindi quali sono le vostre etichette e quali sono i vostri dati di input. 
Quindi in questo caso i nostri dati di input sono immagini e le nostre etichette sono categorie. 
Quindi la categoria sarà o grizzly, o nera, o teddy. 


Quindi questa è la prima cosa che si dice. 
Questo è il parametro del blocco. 
E poi la racconti - come si fa ad avere una lista di tutti i nomi dei file, in questo caso, giusto. 
E abbiamo visto come farlo, perché abbiamo appena chiamato noi stessi la funzione. 
La funzione si chiama get_image_files. 
Quindi le diciamo quale funzione usare per ottenere quella lista di elementi e poi glielo diciamo - come si dividono i dati in un set di validazione e un set di formazione. 
E quindi useremo una cosa chiamata RandomSplitter che lo divide in modo casuale. 
E ne punteremo il 30% nel set di validazione. 
Imposteremo anche il seme casuale che assicura che ogni volta che lo eseguiremo, il set di validazione sarà lo stesso. 
E poi si dice: "Ok, come si etichettano i dati? 
E questo è il nome di una funzione chiamata parent_label. 
E quindi questo cercherà ogni elemento al nome del genitore. 
Quindi questo, questo in particolare diventerebbe un orso nero. 
Ora, questo è come il modo più comune per i dataset di immagini per essere rappresentati, è che essi vengono messi le diverse immagini ottenere i file vengono messi in cartella in base alla loro etichetta. 
E poi finalmente qui abbiamo qualcosa chiamato item_tfms. 
Impareremo molto di più sulle trasformazioni in un attimo. 
Che queste sono fondamentalmente funzioni che vengono applicate ad ogni immagine. 
E così ogni immagine verrà ridimensionata a 128 per 128 quadrati. 
Quindi impareremo presto molto di più sulle API DataBlock. 
Ma fondamentalmente il processo sarà -- chiamerà qualsiasi cosa sia get_items, che è una lista di file di immagini. 


E poi chiamerà get_x, get_y quindi in questo caso non c'è nessun get_x ma c'è un get_y quindi è solo l'etichetta del genitore. 
E poi chiamerà il metodo di creazione per ognuna di queste due cose - creerà un'immagine e creerà una categoria. 
E quindi chiamerà l'item_tfms, che è il ridimensionamento. 
E poi la prossima cosa che fa è metterlo in qualcosa chiamato data loader. 
Un data loader è qualcosa che prende alcune immagini alla volta (penso che di default sia 64) e le mette tutte in un singolo, si chiama batch. 
Prende solo 64 immagini e le attacca tutte insieme. 
E la ragione per cui lo fa è che poi le mette tutte sulla GPU in una volta sola, in modo da poterle passare tutte al modello attraverso la GPU in una sola volta. 
E questo lascia andare la GPU molto più velocemente, come scopriremo. 
E poi, infine (non ne usiamo qui), possiamo avere qualcosa chiamato "batch transforms", di cui parleremo più avanti. 
E poi da qualche parte nel minerale qui concettualmente c'è lo splitter, che è la cosa che si divide nel set di addestramento e nel set di validazione. 
Quindi questo è un modo super flessibile per dire a fast.ai come lavorare con i dati. 
E così alla fine di questo restituisce un oggetto di tipo DataLoaders. 
Ecco perché chiamiamo sempre queste cose DL, giusto. 
Quindi, DataLoaders ha una validazione e un addestramento DataLoader. 
E un DataLoader, come ho appena detto, è qualcosa che prende un lotto di pochi oggetti alla volta e lo mette sulla GPU per voi. 
Quindi questo è fondamentalmente l'intero codice dei DataLoader. 
Quindi i dettagli non hanno importanza, volevo solo sottolineare che, come molti di questi concetti in fast.ai, quando si guarda cosa sono in realtà, sono piccole cose incredibilmente semplici. 
È letteralmente qualcosa a cui basta passare in pochi caricatori di dati e li memorizza in un attributo. 
E passa e ti restituisce il primo come .treno e il secondo come .valido. 
Così possiamo creare i nostri DataLoaders creando prima di tutto il DataBlock, e poi chiamiamo i DataLoaders, passando nel nostro percorso per creare DL. 


E poi si può chiamare show_batch su questo. 
Potete chiamare show_batch praticamente qualsiasi cosa in fast.ai per vedere i vostri dati. 
E guarda, abbiamo dei grizzly, abbiamo un orsacchiotto, abbiamo un grizzly. 
Quindi l'idea è giusta. 
Io guarderò questi diversi, guarderò l'aumento dei dati la prossima settimana, quindi, salterò l'aumento dei dati e salterò direttamente allo scambio del vostro modello. 
Così, una volta che abbiamo i DL, possiamo, come nella Lezione 1, chiamare cnn_learner per creare un ResNet. 
Questa volta creeremo una ResNet più piccola, una ResNet18. 
Di nuovo, chiedendo il tasso di errore, possiamo poi chiamare di nuovo .fine_tune. 
Così si vede che sono tutte le stesse linee di codice che abbiamo già visto. 
E si vede che il nostro tasso di errore scende da nove a uno, quindi abbiamo un errore dell'1% e dopo l'addestramento per circa 25 secondi. 
Quindi si vede che abbiamo solo 450 immagini che abbiamo allenato per meno di un minuto e che abbiamo solo la matrice di confusione, così possiamo dire: "Voglio creare una classe di interpretazione di classificazione; voglio guardare la matrice di confusione" e la matrice di confusione. 
Come potete vedere, è qualcosa che dice "per le cose che in realtà sono orsi neri, quanti sono previsti essere orsi neri contro orsi grizzly contro orsi di peluche?  
Quindi, le diagonali sono quelle corrette e quindi sembra che ci siano due errori. 
Abbiamo un grizzly che si prevedeva fosse nero e un nero che si prevedeva fosse grizzly. 
Un metodo super, super utile è "tracciare le perdite superiori" e questo mi mostrerà come sono effettivamente i miei errori. 
Quindi, questo qui era stato previsto che fosse un grizzly, ma l'etichetta era "orso nero". 
Questo era quello che si prevedeva fosse un orso nero e l'etichetta era "orso grizzly". 
Questi qui non sono in realtà sbagliati. 
Questo è quello che si prevede sia "nero" ed è in realtà nero. 
Ma, la ragione per cui appaiono in questo è che questi sono quelli che il modello era il meno sicuro di sé. 


Ok, quindi guarderemo il pulitore di classificazione delle immagini la prossima settimana. 
Concentriamoci sul modo in cui lo mettiamo in produzione. 
Quindi, per metterlo in produzione, dobbiamo esportare il modello. 
Quindi, quello che fa l'esportazione del modello è che crea un nuovo file, che di default si chiama "export.pkl", che contiene l'architettura e tutti i parametri del modello. 
Quindi, questo è ora qualcosa che si può copiare su un server da qualche parte e trattarlo come un programma predefinito, giusto? 
Quindi, allora il processo di utilizzo del vostro modello addestrato su nuovi tipi di dati in produzione si chiama "inferenza". 
Quindi, qui ho creato un inference learner caricandolo di nuovo, va bene, e quindi ovviamente non ha senso farlo subito dopo averlo salvato in un notebook. 
Ma, vi sto solo mostrando come funzionerebbe bene. 
Quindi, questa è una cosa che faresti sul tuo server - il che è un'illazione. 
Ricordate che una volta che avete addestrato un modello, potete trattarlo come un programma - potete passargli degli input. 
Quindi, questo è ora il nostro programma. 
Questo è il nostro predittore di orsi. 
Quindi, ora posso chiamarlo "predittore" e posso passargli un'immagine e mi dirà - qui è sicuro al 99,999% - che questo è un "grizzly". 
Quindi, penso che quello che faremo qui è avvolgerlo qui e la prossima settimana finiremo di creare una vera e propria interfaccia grafica per il nostro classificatore di orsi. 
Mostreremo come farlo funzionare gratuitamente su un servizio chiamato "Binder" e, sì, e poi penso che saremo pronti ad immergerci in alcuni dettagli di quello che succede dietro le quinte. 
Qualche domanda o altro prima di concludere, Rachel?  No. 
Ok, perfetto. 
Va bene, grazie a tutti. 
Quindi, speriamo, sì, penso che da qui in poi abbiamo coperto, sapete, la maggior parte delle cose fondamentali dal punto di vista dell'apprendimento automatico che dovremo coprire. 
Quindi, saremo in grado, pronti ad immergerci in dettagli di basso livello su come funziona l'apprendimento profondo dietro le quinte e penso che questo inizierà dalla prossima settimana. 
Allora, ci vediamo.