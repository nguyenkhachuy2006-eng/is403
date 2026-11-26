"""Chuyển ``CONFIG`` / ``PLOT`` thành object dùng cho toàn pipeline."""

from lstnet_config import CONFIG, PLOT


class LSTNetInit(object):
    """Object cấu hình chỉ đọc, được dựng từ ``CONFIG`` và ``PLOT``."""

    def __init__(self, config=None, plot=None):
        config = CONFIG if config is None else config
        plot = PLOT if plot is None else plot

        # --- Dữ liệu ---
        self.data = config["data"]
        self.date_col = config["date_col"]
        self.price_col = config["price_col"]
        self.target_col = config["target_col"]
        self.features = list(config["features"])
        self.train_ratio = float(config["train_ratio"])
        self.fit_ratio = float(config["fit_ratio"])
        self.window = int(config["window"])
        self.scaler = config["scaler"]

        # --- Kiến trúc ---
        self.CNNFilters = int(config["CNNFilters"])
        self.CNNKernel = int(config["CNNKernel"])
        self.GRUUnits = int(config["GRUUnits"])
        self.SkipGRUUnits = int(config["SkipGRUUnits"])
        self.skip = int(config["skip"])
        self.dropout = float(config["dropout"])
        self.highway = int(config["highway"])
        self.initialiser = config["initializer"]

        # --- Huấn luyện ---
        self.optimiser = config["optimizer"]
        self.lr = float(config["lr"])
        self.loss = config["loss"]
        self.batchsize = int(config["batchsize"])
        self.epochs = int(config["epochs"])
        self.earlystop_patience = int(config["earlystop_patience"])
        self.earlystop_min_delta = float(config["earlystop_min_delta"])

        # --- Tái lập & đầu ra ---
        self.seeds = list(config["seeds"])
        self.pred_dir = config["pred_dir"]

        # --- Biểu đồ ---
        self.show_history = bool(plot["history"])
        self.show_prediction = bool(plot["prediction"])
        self.save_plot = plot["save_plot"]
        self.dpi = int(plot["dpi"])

        self._validate()

    def _validate(self):
        if self.window < 2:
            raise ValueError("window phải >= 2")
        if self.highway > self.window - 1:
            raise ValueError(
                "highway (%d) phải <= window - 1 (%d) để có đủ delta quá khứ"
                % (self.highway, self.window - 1)
            )
        if self.highway < 0:
            raise ValueError("highway phải >= 0")
        if not (0 < self.train_ratio < 1):
            raise ValueError("train_ratio phải nằm trong (0, 1)")
        if not (0 < self.fit_ratio < 1):
            raise ValueError("fit_ratio phải nằm trong (0, 1)")
        if not self.seeds:
            raise ValueError("seeds không được rỗng")

    def dump(self):
        for key, value in vars(self).items():
            print("%-20s: %s" % (key, value))
