#!/usr/bin/env python3
"""
钓鱼URL分类系统增强版
功能：
1. 支持生成超大规模数据集（10万+样本）
2. 交互式钓鱼模拟器
3. 包含传统ML和深度学习模型
4. 实时预测功能
"""

import os
import json
import random
import pandas as pd
import numpy as np
import requests
import tldextract
from urllib.parse import urlparse
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt


class PhishingDatasetGenerator:
    def __init__(self, output_dir="phishing_data"):
        """初始化数据集生成器"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 日志系统
        self.log_file = os.path.join(output_dir, "generator.log")
        self._init_log()

        # 预定义数据
        self.legit_domains = self._load_top_domains()
        self.phish_keywords = self._load_phish_keywords()

    def _init_log(self):
        """初始化日志"""
        with open(self.log_file, "w") as f:
            f.write(f"钓鱼数据集生成器 {datetime.now()}\n")
            f.write("=" * 50 + "\n")

    def log(self, message):
        """记录日志"""
        with open(self.log_file, "a") as f:
            f.write(f"[{datetime.now()}] {message}\n")
        print(message)

    def _load_top_domains(self, count=1000):
        """加载全球Top域名"""
        try:
            url = "https://www.domcop.com/files/top/top10milliondomains.csv.zip"
            resp = requests.get(url, timeout=10)
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(f)
            return df['Domain'].head(count).tolist()
        except:
            # 备用列表
            return ["google.com", "youtube.com", "facebook.com", "baidu.com",
                    "wikipedia.org", "qq.com", "taobao.com", "amazon.com",
                    "yahoo.com", "tmall.com", "twitter.com", "sohu.com"]

    def _load_phish_keywords(self):
        """加载钓鱼关键词"""
        return ["login", "verify", "account", "secure", "banking",
                "update", "confirm", "password", "paypal", "alert"]

    def generate_dataset(self, total_size=100000):
        """生成大规模数据集"""
        self.log(f"开始生成数据集，目标大小: {total_size:,}")

        # 生成合法URL (60%)
        legit_urls = self._generate_legit_urls(int(total_size * 0.6))

        # 生成钓鱼URL (40%)
        phish_urls = self._generate_phish_urls(int(total_size * 0.4))

        # 合并数据集
        df = pd.DataFrame({
            "url": legit_urls + phish_urls,
            "label": [0] * len(legit_urls) + [1] * len(phish_urls)
        })

        # 打乱顺序
        df = df.sample(frac=1).reset_index(drop=True)

        # 保存数据集
        csv_path = os.path.join(self.output_dir, "phishing_dataset.csv")
        df.to_csv(csv_path, index=False)
        self.log(f"数据集已保存到 {csv_path} (总样本: {len(df):,})")

        return df

    def _generate_legit_urls(self, count):
        """生成合法URL"""
        paths = ["", "home", "search", "products", "about", "contact",
                 "news", "blog", "faq", "terms"]

        urls = []
        for _ in range(count):
            domain = random.choice(self.legit_domains)
            path = random.choice(paths)
            urls.append(f"https://{domain}/{path}")
        return urls

    def _generate_phish_urls(self, count):
        """生成钓鱼URL"""
        patterns = [
            "{keyword}-{domain}",
            "{domain}-{keyword}",
            "secure-{domain}",
            "{domain}-login",
            "update-{domain}"
        ]

        urls = []
        for _ in range(count):
            domain = random.choice(self.legit_domains)
            keyword = random.choice(self.phish_keywords)
            pattern = random.choice(patterns)

            fake_domain = pattern.format(domain=domain, keyword=keyword)
            urls.append(f"http://{fake_domain}/login.php")
        return urls


class PhishingClassifier:
    def __init__(self, data_path):
        """初始化分类器"""
        self.df = pd.read_csv(data_path)
        self.models = {}
        self.tokenizer = Tokenizer(num_words=10000)

    def prepare_data(self):
        """准备训练数据"""
        # 特征工程
        self._extract_features()

        # 文本特征
        self._process_text_features()

        # 拆分数据集
        X = np.hstack([
            self.features.values,
            self.tfidf_features.toarray(),
            self.char_features.toarray()
        ])
        y = self.df['label'].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # 准备LSTM数据
        self._prepare_sequences()

    def _extract_features(self):
        """提取结构化特征"""
        features = []
        for url in self.df['url']:
            parsed = urlparse(url)
            ext = tldextract.extract(url)

            features.append({
                "url_len": len(url),
                "domain_len": len(parsed.netloc),
                "num_dots": url.count('.'),
                "num_hyphens": url.count('-'),
                "has_https": int(parsed.scheme == 'https'),
                "tld": ext.suffix,
                "subdomain_len": len(ext.subdomain),
                "path_len": len(parsed.path),
                "query_len": len(parsed.query),
                "has_ip": int(self._is_ip(parsed.netloc))
            })

        self.features = pd.DataFrame(features)
        self.features['tld'] = self.features['tld'].astype('category').cat.codes

    def _process_text_features(self):
        """处理文本特征"""
        # 词级特征
        self.tfidf_vectorizer = TfidfVectorizer(max_features=500)
        self.tfidf_features = self.tfidf_vectorizer.fit_transform(self.df['url'])

        # 字符级特征
        self.char_vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 5), max_features=500)
        self.char_features = self.char_vectorizer.fit_transform(self.df['url'])

    def _prepare_sequences(self):
        """准备LSTM序列数据"""
        self.tokenizer.fit_on_texts(self.df['url'])
        sequences = self.tokenizer.texts_to_sequences(self.df['url'])
        self.max_len = max(len(x) for x in sequences)
        self.sequences = pad_sequences(sequences, maxlen=self.max_len)

        # 拆分LSTM数据
        X = self.sequences
        y = self.df['label'].values
        self.X_train_lstm, self.X_test_lstm, self.y_train_lstm, self.y_test_lstm = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

    def train_models(self):
        """训练所有模型"""
        # 随机森林
        self._train_rf()

        # LSTM
        self._train_lstm()

    def _train_rf(self):
        """训练随机森林"""
        self.log("训练随机森林...")
        self.rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        self.rf.fit(self.X_train, self.y_train)

        # 评估
        y_pred = self.rf.predict(self.X_test)
        report = classification_report(self.y_test, y_pred)
        self.log("随机森林性能:\n" + report)

        self.models['rf'] = self.rf

    def _train_lstm(self):
        """训练LSTM模型"""
        self.log("训练LSTM...")
        vocab_size = len(self.tokenizer.word_index) + 1

        model = Sequential([
            Embedding(vocab_size, 64, input_length=self.max_len),
            LSTM(64, dropout=0.2),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        history = model.fit(
            self.X_train_lstm, self.y_train_lstm,
            validation_data=(self.X_test_lstm, self.y_test_lstm),
            epochs=5,
            batch_size=128
        )

        self.models['lstm'] = model
        self._plot_history(history)

    def _plot_history(self, history):
        """绘制训练历史"""
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title('LSTM Training History')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "lstm_history.png"))
        plt.close()

    def predict(self, url, model_type='rf'):
        """预测URL类型"""
        if model_type == 'rf':
            # 提取特征
            features = self._extract_single_features(url)
            tfidf = self.tfidf_vectorizer.transform([url])
            char = self.char_vectorizer.transform([url])

            X = np.hstack([
                features.values,
                tfidf.toarray(),
                char.toarray()
            ])

            proba = self.rf.predict_proba(X)[0]
            pred = self.rf.predict(X)[0]
        else:
            # LSTM预测
            seq = self.tokenizer.texts_to_sequences([url])
            padded = pad_sequences(seq, maxlen=self.max_len)

            proba = self.models['lstm'].predict(padded)[0][0]
            pred = 1 if proba > 0.5 else 0
            proba = [1 - proba, proba]

        return {
            "url": url,
            "prediction": "钓鱼" if pred == 1 else "合法",
            "phishing_prob": proba[1],
            "legitimate_prob": proba[0],
            "model": model_type.upper()
        }


class PhishingSimulator:
    def __init__(self, classifier):
        """初始化模拟器"""
        self.classifier = classifier
        self.score = 0
        self.total = 0

    def start(self, rounds=10):
        """启动交互式模拟"""
        print("\n" + "=" * 50)
        print("钓鱼URL识别模拟器")
        print(f"将测试 {rounds} 个URL，请判断是否为钓鱼网站")
        print("=" * 50 + "\n")

        for i in range(1, rounds + 1):
            self._run_round(i)

        self._show_summary()

    def _run_round(self, round_num):
        """运行单轮测试"""
        # 随机选择一个URL
        row = self.classifier.df.sample(1).iloc[0]
        url = row['url']
        actual = row['label']

        print(f"\n[第 {round_num} 轮] URL: {url}")
        print("-" * 50)

        # 显示特征分析
        self._show_features(url)

        # 获取用户输入
        while True:
            guess = input("你认为这是钓鱼网站吗? (y/n): ").lower()
            if guess in ['y', 'n']:
                break
            print("请输入 y 或 n")

        # 验证答案
        user_pred = 1 if guess == 'y' else 0
        is_correct = user_pred == actual

        # 更新分数
        self.total += 1
        if is_correct:
            self.score += 1

        # 显示结果
        result = "正确" if is_correct else "错误"
        print(f"\n结果: {result}")
        print(f"实际类别: {'钓鱼' if actual == 1 else '合法'}")

        # 显示模型预测
        model_pred = self.classifier.predict(url)
        print(f"\nAI预测 ({model_pred['model']}):")
        print(f"- 预测: {model_pred['prediction']}")
        print(f"- 可信度: {model_pred['phishing_prob']:.2%}")

        input("\n按回车继续...")

    def _show_features(self, url):
        """显示URL特征"""
        parsed = urlparse(url)
        ext = tldextract.extract(url)

        print("\nURL分析:")
        print(f"- 协议: {parsed.scheme}")
        print(f"- 域名: {parsed.netloc}")
        print(f"- 路径: {parsed.path}")
        print(f"- 查询: {parsed.query}")
        print(f"- 子域名: {ext.subdomain}")
        print(f"- 主域名: {ext.domain}")
        print(f"- TLD: {ext.suffix}")

        # 显示危险信号
        danger_signs = 0
        if parsed.scheme != 'https':
            print("⚠️ 使用不安全的HTTP协议")
            danger_signs += 1
        if '-' in ext.domain:
            print("⚠️ 域名包含连字符")
            danger_signs += 1
        if len(parsed.path.split('/')) > 3:
            print("⚠️ 路径层级过深")
            danger_signs += 1

        print(f"检测到 {danger_signs} 个危险信号")

    def _show_summary(self):
        """显示总结"""
        print("\n" + "=" * 50)
        print("模拟结束!")
        print(f"最终得分: {self.score}/{self.total}")
        print(f"准确率: {self.score / self.total:.1%}")
        print("=" * 50 + "\n")


def main():
    """主程序"""
    print("钓鱼URL分类系统 v2.0")
    print("=" * 50)

    # 初始化
    data_dir = "phishing_data"
    data_path = os.path.join(data_dir, "phishing_dataset.csv")

    # 生成数据集（如果不存在）
    if not os.path.exists(data_path):
        generator = PhishingDatasetGenerator(data_dir)
        generator.generate_dataset(total_size=100000)

    # 训练分类器
    classifier = PhishingClassifier(data_path)
    classifier.prepare_data()
    classifier.train_models()

    # 启动交互式模拟
    simulator = PhishingSimulator(classifier)

    while True:
        print("\n菜单:")
        print("1. 开始交互式模拟")
        print("2. 测试单个URL")
        print("3. 退出")

        choice = input("请选择操作: ")

        if choice == '1':
            rounds = int(input("请输入测试轮数: "))
            simulator.start(rounds)
        elif choice == '2':
            url = input("请输入要检测的URL: ")
            result = classifier.predict(url)
            print("\n检测结果:")
            print(f"- 预测: {result['prediction']}")
            print(f"- 模型: {result['model']}")
            print(f"- 钓鱼概率: {result['phishing_prob']:.2%}")
        elif choice == '3':
            break
        else:
            print("无效输入，请重新选择")


if __name__ == "__main__":
    main()