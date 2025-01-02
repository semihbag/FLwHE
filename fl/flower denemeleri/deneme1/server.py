import flwr as fl

# Federated Learning Server'ı başlat
if __name__ == "__main__":
    # Sunucuyu başlat
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),  # Toplam 3 tur
    )
