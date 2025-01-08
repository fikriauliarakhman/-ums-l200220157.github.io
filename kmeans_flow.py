# from metaflow import FlowSpec, step, Parameter
# 
# class KmeansFlow(FlowSpec):
    # num_docs = Parameter('num-docs', help='Number of documents', default=1000)
# 
    # @step
    # def start(self):
        # import scale_data #A
        # scale_data.load_yelp_riviews(self.num_docs) #A
        # self.next(self.end)
# 
    # @step
    # def end(self):
        # pass
# 
# if __name__ == '__main__':
    # KmeansFlow()
# 
#A Impor modul yang kita buat sebelumnya dan gunakan untuk memuat dataset

from metaflow import FlowSpec, step, Parameter

class KmeansFlow(FlowSpec):
    num_docs = Parameter('num-docs', help='Number of documents', default=1000)

    @step
    def start(self):
        import scale_data
        # Load WhatsApp data
        self.data = scale_data.load_whatsapp_data(self.num_docs)
        self.next(self.vectorize)

    @step
    def vectorize(self):
        # Convert the data into a document-term matrix
        import scale_data
        docs = [row[1] for row in self.data]  # Assuming the second column contains the text
        self.matrix, self.cols = scale_data.make_matrix(docs)
        self.next(self.cluster)

    @step
    def cluster(self):
        # Apply KMeans clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        self.clusters = kmeans.fit_predict(self.matrix)
        self.next(self.end)

    @step
    def end(self):
        print("Clustering completed!")
        print("Top terms per cluster:")
        from collections import Counter
        for cluster_id in range(3):
            cluster_docs = [self.cols[idx] for idx, label in enumerate(self.clusters) if label == cluster_id]
            top_terms = Counter(cluster_docs).most_common(3)
            print(f"Cluster {cluster_id}: {top_terms}")

if __name__ == '__main__':
    KmeansFlow()

# load_whatsapp_data: Mengganti fungsi untuk membaca file lokal (cleaned_data.csv).
# vectorize step: Menambahkan langkah untuk memproses teks menjadi matriks.
# cluster step: Menjalankan algoritma KMeans pada matriks dokumen.
# end step: Menampilkan 3 kata teratas dalam setiap klaster.
