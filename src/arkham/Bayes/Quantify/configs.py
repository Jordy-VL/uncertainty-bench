import os
from sacred import Experiment
from arkham.default_config import MODELROOT, SAVEROOT, DATAROOT

from arkham.utils.utils import generate_out_folder, remove_progress
from arkham.Bayes.Quantify.data import generate_char2idx_idx2char


DEFAULT_EXPERIMENT = "quantify_textCNN"
ex = Experiment(DEFAULT_EXPERIMENT)
ex.captured_out_filter = remove_progress


@ex.config
def default():
    """
    this default config is used to show typing of arguments and appropriate options
    it is required by Sacred as it would otherwise constitute unused variables
    """
    identifier = "mini_imdb"
    data_folder = os.path.join(DATAROOT, identifier)
    out_folder = generate_out_folder(data_folder)
    task = "document_classification"  # or regression
    column_map = None

    downsampling = 0
    buffer_size = 1000
    sort_by_seqlen = True
    batch_size = 32

    composition = ["word"]  # ["sentence", "word", "character", "casing"]
    token_pattern = r"\b\w\S+\b"
    lowercase = True if "casing" in composition else False
    min_token_len = 2
    min_token_freq = 2
    max_vocabulary = 0  # can be int OR proportion of total; 0 means use the whole vocabulary
    max_document_len = None
    max_sentences = None
    raw = False

    model = None
    pretrained_embeddings = None
    overblend_vocabulary = True  # if not "character" in composition else False
    embed_dim = 100
    kernel_sizes = []  # [1,2,3] or [2,3,4]
    feature_maps = []
    projection_nodes = 0
    dropout = 0
    dropout_concrete = None
    dropout_nonlinear = 0
    embedding_dropout = 0
    weight_decay = None
    batchnorm = False

    num_heads = 2
    model_class = ""
    finetune = False

    # CHAR-level
    use_char = True if "character" in composition else False
    char_voc = generate_char2idx_idx2char(as_list=True) if use_char else []
    max_chars_len = 200
    char_embed_dim = 30
    char_kernel_sizes = [3, 10, 20]
    char_feature_maps = [100, 100, 100]

    epochs = 2
    learning_rate = 0.001
    steps_per_epoch = None
    optimizer = "adam"
    clipnorm = 10
    gamma = 3
    label_smoothing = 0
    loss_fn = "categorical_crossentropy"
    metrics = ["accuracy", "ECE"]

    calibration = False  # apply temperature scaling or calibration
    use_aleatorics = False
    posterior_sampling = 10
    ensemble = 1
    ood = None  # ["singleton", "list", "odd"] #uneven
    multilabel = False
    sequence_labels = False
    pad_value = None

    use_gp_layer = False
    spec_norm_multipliers = [1, 2]

    alpha = 1
    cycles = 1


@ex.named_config
def clf_default():
    identifier = "imdb"
    data_folder = os.path.join(DATAROOT, identifier)
    out_folder = generate_out_folder(data_folder)
    task = "document_classification"  # or regression
    max_vocabulary = 20000  # 20news to 30K e.g.
    composition = ["word"]

    model = "TextClassificationCNN_simple"
    embed_dim = 300
    kernel_sizes = [3, 4, 5]
    feature_maps = [100, 100, 100]
    projection_nodes = 100
    dropout = 0.5
    dropout_nonlinear = 0.5  # activated by default!
    embedding_dropout = 0  # should be activated by default; to be verified
    dropout_concrete = None
    weight_decay = 0.0001  # triggers AdamW optimizer

    epochs = 48
    optimizer = "adam"
    clipnorm = 10
    learning_rate = 0.001
    steps_per_epoch = 2000  # could also be None

    posterior_sampling = 10
    use_aleatorics = False  # can use numeric for output_layer
    multilabel = True if identifier in ["AAPD", "Reuters_multilabel"] else False
    loss_fn = (
        "categorical_crossentropy"
        if not use_aleatorics
        else "attenuated_learned_loss"
        if not multilabel
        else "attenuated_learned_loss_multilabel"
    )
    metrics = ["accuracy", "mse", "ECE"] if not use_aleatorics else ["accuracy"]  # DEV: ["heteroscedastic_mse"]


@ex.named_config
def SNGP_default():
    model = "TextClassificationCNN_SNGP"
    loss_fn = "categorical_crossentropy_from_logits"
    use_gp_layer = True
    weight_decay = 0
    spec_norm_multipliers = [1, 2]  # Dense - Conv
    metrics = ["accuracy"]


@ex.named_config
def cSGMCMC_clf():
    optimizer = "cSGLD"
    alpha = 0.95
    cycles = 4
    weight_decay = 0
    learning_rate = 0.1  # to start aggressive exploration
    posterior_sampling = 3  # models to save per cycle
    epochs = 100  # budget to fit many cycles


@ex.named_config
def conll_default():
    # https://arxiv.org/pdf/1603.01360v3.pdf
    identifier = "conll_03"
    data_folder = os.path.join(DATAROOT, identifier)
    out_folder = generate_out_folder(data_folder)
    task = "entity_extraction"

    downsampling = 0
    buffer_size = 1000
    batch_size = 32

    token_pattern = ""
    lowercase = False
    min_token_len = 0
    min_token_freq = 0
    max_vocabulary = 0
    max_document_len = None
    max_sentences = None
    composition = ["word"]
    embed_dim = 100

    model = "sequential_bilstm"
    projection_nodes = 100
    pretrained_embeddings = "/mnt/lerna/embeddings/glove.6B.50d.txt"
    dropout = 0.25
    dropout_nonlinear = 0.5
    epochs = 40
    optimizer = "adam"
    weight_decay = 0  # 0.00001
    learning_rate = 0.001  # 0.005 #0.01
    clipnorm = None  # 5

    posterior_sampling = 10
    use_aleatorics = False
    multilabel = False
    sequence_labels = True
    loss_fn = "sparse_categorical_crossentropy" if not use_aleatorics else "sequential_attenuated_learned_loss"
    metrics = ["accuracy"] if not use_aleatorics else None


@ex.named_config
def charbilstm_NER():
    # SOURCE: https://github.com/mxhofer/Named-Entity-Recognition-BidirectionalLSTM-CNN-CoNLL/blob/master/nn_CoNLL.ipynb
    # https://www.depends-on-the-definition.com/lstm-with-char-embeddings-for-ner/
    # https://arxiv.org/pdf/1603.01360v3.pdf
    # https://arxiv.org/pdf/1511.08308.pdf
    identifier = "conll_03"
    data_folder = os.path.join(DATAROOT, identifier)
    out_folder = generate_out_folder(data_folder)
    task = "entity_extraction"

    downsampling = 0
    buffer_size = 1000
    batch_size = 32
    sort_by_seqlen = True

    composition = ["word", "character", "casing"]  # could add "case"
    lowercase = True if "casing" in composition else False

    token_pattern = ""
    min_token_len = 0
    min_token_freq = 0
    max_vocabulary = 0
    max_document_len = None
    max_sentences = None

    # CHAR-level
    use_char = True if "character" in composition else False
    char_voc = generate_char2idx_idx2char(as_list=True) if use_char else []
    max_chars_len = 52  # how long is the max tokenized word; could lower
    char_embed_dim = 25
    char_kernel_sizes = [3]
    char_feature_maps = [50]

    model = "sequential_bilstm"
    embed_dim = 200
    pretrained_embeddings = "/mnt/lerna/embeddings/glove.6B.50d.txt"
    overblend_vocabulary = True if not "character" in composition else False
    projection_nodes = 200
    dropout = 0.5  # input
    dropout_nonlinear = 0.25  # recurrent
    epochs = 80  # 30
    optimizer = "adam"
    weight_decay = 0  # 0.00001
    learning_rate = 0.00105  # 0.005 #0.01
    clipnorm = None  # 5

    posterior_sampling = 10
    use_aleatorics = False
    multilabel = False
    sequence_labels = True
    loss_fn = "sparse_categorical_crossentropy" if not use_aleatorics else "sequential_attenuated_learned_loss"
    metrics = ["accuracy"] if not use_aleatorics else None
    pad_value = None


@ex.named_config
def BERT_NER():
    # SOURCE: https://androidkt.com/name-entity-recognition-with-bert-in-tensorflow/
    identifier = "conll_03"
    data_folder = os.path.join(DATAROOT, identifier)
    out_folder = generate_out_folder(data_folder)
    task = "entity_extraction"

    downsampling = 0
    buffer_size = 1000
    batch_size = 4
    sort_by_seqlen = True

    composition = ["word"]  # could add "case"
    lowercase = True if "casing" in composition else False

    token_pattern = ""
    min_token_len = 0
    min_token_freq = 0
    max_vocabulary = 0
    max_document_len = 512
    max_sentences = None

    model = "BERT_dense"
    model_class = "bert-base-cased"
    finetune = False
    projection_nodes = 200
    learning_rate = 0.00003
    epochs = 5  # 30

    dropout = 0.1  # input
    dropout_nonlinear = 0.2  # recurrent
    optimizer = "adam"
    weight_decay = 0  # 0.00001
    clipnorm = None  # 5
    use_aleatorics = False
    multilabel = False
    sequence_labels = True
    loss_fn = "sparse_crossentropy_masked_v2"  # SparseMaskCatCELoss
    metrics = ["sparse_categorical_accuracy_masked", "ECE"] if not use_aleatorics else None
    pad_value = -100


@ex.named_config
def focal_loss_config():
    """
    Focal loss (Lin et al., 2017) (FL):  −α_t (1-p_t)^y log(p_t)
    loss = (1 - probs) ** gamma * xent_loss
    originally improving Average Precision (AP) in single-stage object detectors.

    Label smoothing [Müller et al., 2019] (LS): given a one-hot ground-truth distribution q and a smoothing factor α (hyperparameter),
    the smoothed vector s is obtained as si = (1 − α)qi + α(1 − qi )/(K − 1)

    Gamma adaptive: sample-dependent schedule FLSD-53; (γ = 5 for pˆy ∈ [0, 0.2), and γ = 3 for pˆy ∈ [0.2, 1] [https://arxiv.org/pdf/2002.09437.pdf]
    """
    loss_fn = "sparse_focal_loss"
    gamma = 3
    label_smoothing = 0


@ex.named_config
def BERT_dummy():
    # SOURCE: https://androidkt.com/name-entity-recognition-with-bert-in-tensorflow/
    identifier = "conll_03"
    data_folder = os.path.join(DATAROOT, identifier)
    out_folder = generate_out_folder(data_folder)
    task = "entity_extraction"

    downsampling = 0.01
    buffer_size = 1000
    batch_size = 4
    sort_by_seqlen = True

    composition = ["word"]  # could add "case"
    lowercase = True if "casing" in composition else False

    token_pattern = ""
    min_token_len = 0
    min_token_freq = 0
    max_vocabulary = 0
    max_document_len = 100
    max_sentences = None

    model = "BERT_dense"
    model_class = "bert-base-cased"
    finetune = False
    projection_nodes = 200
    learning_rate = 0.00003
    epochs = 2  # 30

    dropout = 0.1  # input
    dropout_nonlinear = 0.2  # recurrent
    optimizer = "adam"
    weight_decay = 0  # 0.00001
    clipnorm = None  # 5
    use_aleatorics = False
    multilabel = False
    sequence_labels = True
    loss_fn = "sparse_crossentropy_masked_v2"  # "sparse_categorical_crossentropy" if not use_aleatorics else "sequential_attenuated_learned_loss"
    metrics = ["sparse_categorical_accuracy_masked"] if not use_aleatorics else None


@ex.named_config
def ovadia_20news():
    """
    %maxdoclen 250; max_vocabulary 30000
    %The vanilla model uses a one-layer LSTM model of size 32 and a dense layer to predict the 10 class
    % probabilities based on word embedding of size 128. A dropout rate of 0.1 is applied to both the LSTM
    % layer and the dense layer for the Dropout model. The LL-SVI model replaces the last dense layer
    % with a Bayesian layer, the ensemble model aggregates 10 vanilla models, and stochastic methods
    % sample 5 predictions per example. The vanilla model accuracy for in-distribution test data is 0.955.
    """
    identifier = "20news"
    data_folder = os.path.join(DATAROOT, identifier)
    out_folder = generate_out_folder(data_folder)
    task = "document_classification"  # or regression
    max_vocabulary = 30000  # 20news to 30K e.g.
    composition = ["word"]

    model = "lstm"
    embed_dim = 128
    projection_nodes = 32
    dropout = 0.1
    dropout_nonlinear = 0.1
    embedding_dropout = 0
    dropout_concrete = None
    weight_decay = 0  # .0001  # triggers AdamW optimizer
    max_document_len = 250

    epochs = 48
    optimizer = "adam"
    clipnorm = 10
    learning_rate = 0.001
    steps_per_epoch = None  # could also be None

    posterior_sampling = 10
    use_aleatorics = False
    multilabel = False
    loss_fn = "categorical_crossentropy" if not use_aleatorics else "attenuated_learned_loss"
    metrics = ["accuracy", "mse"] if not use_aleatorics else []
    ood = [
        'comp.graphics',
        'comp.sys.ibm.pc.hardware',
        'comp.windows.x',
        'rec.autos',
        'rec.sport.baseball',
        'sci.crypt',
        'sci.med',
        'soc.religion.christian',
        'talk.politics.mideast',
        'talk.religion.misc',
    ]
