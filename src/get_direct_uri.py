#!/usr/bin/env python3
"""
get_direct_uri.py — Convert your mongodb+srv:// URI to a direct mongodb:// URI.

Run this ONCE on a machine with working DNS (or use the Atlas UI),
then paste the output into your .env as MONGODB_URI.

Usage:
    python get_direct_uri.py
    python get_direct_uri.py --uri "mongodb+srv://user:pass@cluster.mongodb.net/"
"""
import sys
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--uri", default=None)
parser.add_argument("--env", default=None)
args = parser.parse_args()

# Load from .env if no --uri given
uri = args.uri
if not uri:
    env_path = Path(args.env) if args.env else Path(__file__).parent / ".env"
    if env_path.exists():
        for line in open(env_path):
            line = line.strip()
            if line.startswith("MONGODB_URI="):
                uri = line[len("MONGODB_URI="):].strip().strip('"').strip("'")
                break

if not uri:
    print("ERROR: No URI found. Pass --uri or ensure MONGODB_URI is in .env")
    sys.exit(1)

if not uri.startswith("mongodb+srv://"):
    print(f"URI is already a standard mongodb:// — no conversion needed:\n{uri}")
    sys.exit(0)

print(f"Input SRV URI: {uri}\n")

# Parse
after = uri[len("mongodb+srv://"):]
creds = None
if "@" in after:
    creds, rest = after.split("@", 1)
else:
    rest = after
hostname = rest.split("/")[0].split("?")[0]
db_and_opts = rest[len(hostname):]

print(f"Cluster hostname: {hostname}")
print(f"Attempting SRV lookup for: _mongodb._tcp.{hostname}\n")

try:
    import dns.resolver

    # Try multiple nameservers
    for ns in ["8.8.8.8", "1.1.1.1", "9.9.9.9", "208.67.222.222"]:
        try:
            r = dns.resolver.Resolver(configure=False)
            r.nameservers = [ns]
            r.lifetime = 8
            answers = r.resolve(f"_mongodb._tcp.{hostname}", "SRV")
            hosts = ",".join(
                f"{a.target.to_text().rstrip('.')}:{a.port}" for a in answers
            )
            cred_prefix = f"{creds}@" if creds else ""
            direct = f"mongodb://{cred_prefix}{hosts}{db_and_opts}"
            sep = "&" if "?" in direct else "?"
            if "ssl" not in direct and "tls" not in direct:
                direct += f"{sep}ssl=true"; sep = "&"
            if "authSource" not in direct:
                direct += f"{sep}authSource=admin"

            print(f"✓ Resolved via {ns}")
            print(f"\nDirect URI:\n{direct}")
            print(f"\nPaste this into your .env as:\nMONGODB_URI={direct}")
            sys.exit(0)
        except Exception as e:
            print(f"  {ns} failed: {e}")

    print("\nAll DNS servers failed — DNS port 53 is blocked on this machine.")
except ImportError:
    print("dnspython not installed: pip install dnspython")

print("""
─────────────────────────────────────────────────────────
MANUAL STEPS (get the direct URI from Atlas UI):

1. Go to your Atlas cluster → Connect → Drivers
2. Select Python driver
3. Look for the connection string — it should start with mongodb://
   (NOT mongodb+srv://)
   It looks like:
   mongodb://user:pass@shard-00-00.xxxxx.mongodb.net:27017,
             shard-00-01.xxxxx.mongodb.net:27017,
             shard-00-02.xxxxx.mongodb.net:27017/
             ?ssl=true&replicaSet=atlas-xxxxx&authSource=admin

4. Paste that full string into your .env:
   MONGODB_URI=mongodb://user:pass@shard-00-00...

─────────────────────────────────────────────────────────
""")
