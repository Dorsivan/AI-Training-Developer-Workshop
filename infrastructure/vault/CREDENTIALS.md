# Vault Credentials

## Access

- **UI URL**: https://vault-vault.apps.ocp.lgvzs.sandbox180.opentlc.com
- **Method**: Token
- **Root Token**: `hvs.O9bW1PhetRG69QVBZoamz3Kh`
- **Unseal Key**: `RR/mdc8x9sE36xU0LciuRHXo5iycCaLljZgH1nPgF28=`

## Example Secrets

| Path                | Keys                          |
|---------------------|-------------------------------|
| `secret/example`    | username, password            |
| `secret/database`   | host, port, username, password|

## CLI Usage

```bash
export VAULT_ADDR="https://vault-vault.apps.ocp.lgvzs.sandbox180.opentlc.com"
export VAULT_TOKEN="hvs.O9bW1PhetRG69QVBZoamz3Kh"

vault kv get secret/example
vault kv get secret/database
```

## Notes

- Vault is initialized with a single unseal key (key-shares=1, key-threshold=1) for simplicity.
- If the pod restarts, you will need to unseal Vault again using the unseal key above.
- This is a dev/demo setup — do NOT use in production.
