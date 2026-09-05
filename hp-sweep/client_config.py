import os
import kfp

def make_client():
    options = {
        "host": os.environ["SWEEP_API_URL"],
        "existing_token": os.environ["SWEEP_TOKEN"],
        "namespace": os.environ["SWEEP_NAMESPACE"],
    }
    if os.environ.get("SWEEP_CA_CERT"):
        options["ssl_ca_cert"] = os.environ["SWEEP_CA_CERT"]
    return kfp.Client(**options)

if __name__ == "__main__":
    print(make_client().list_experiments())
