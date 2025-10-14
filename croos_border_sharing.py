import json
import time
import hashlib
import hmac
from typing import List, Dict


def canonical_json(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def create_evidence_hash(evidence: dict) -> str:
    return sha256_hex(canonical_json(evidence))


def sign_with_hmac(key: bytes, message: bytes) -> str:
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def verify_hmac_signature(key: bytes, message: bytes, signature_hex: str) -> bool:
    expected = hmac.new(key, message, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_hex)


class Block:
    def __init__(self, index: int, prev_hash: str, evidence_hash: str, metadata: dict, signer_id: str, timestamp: float = None, signature: str = None):
        self.index = index
        self.prev_hash = prev_hash
        self.evidence_hash = evidence_hash
        self.metadata = metadata
        self.signer_id = signer_id
        self.timestamp = timestamp or time.time()
        self.signature = signature

    def header_dict(self) -> dict:
        return {
            "index": self.index,
            "prev_hash": self.prev_hash,
            "evidence_hash": self.evidence_hash,
            "metadata": self.metadata,
            "signer_id": self.signer_id,
            "timestamp": int(self.timestamp),
        }

    def to_json(self) -> dict:
        return {**self.header_dict(), "signature": self.signature}


class SimplePermissionedChain:
    def __init__(self):
        self.chain: List[Block] = []
        genesis = Block(index=0, prev_hash="0" * 64, evidence_hash="0" * 64, metadata={"note": "genesis"}, signer_id="genesis", timestamp=0)
        genesis.signature = "0" * 64
        self.chain.append(genesis)

    def last_hash(self) -> str:
        last = self.chain[-1]
        return sha256_hex(canonical_json(last.header_dict()))

    def add_signed_block(self, evidence_hash: str, metadata: dict, signer_id: str, signer_key: bytes) -> Block:
        idx = len(self.chain)
        prev = self.last_hash()
        block = Block(index=idx, prev_hash=prev, evidence_hash=evidence_hash, metadata=metadata, signer_id=signer_id)
        header = canonical_json(block.header_dict())
        signature = sign_with_hmac(signer_key, header)
        block.signature = signature
        self.chain.append(block)
        return block

    def verify_chain(self, trusted_keys: Dict[str, bytes]) -> bool:
        for i in range(1, len(self.chain)):
            cur = self.chain[i]
            prev = self.chain[i - 1]
            expected_prev_hash = sha256_hex(canonical_json(prev.header_dict()))
            if cur.prev_hash != expected_prev_hash:
                print(f"[Chain verify] BAD prev_hash at index {i}")
                return False
            key = trusted_keys.get(cur.signer_id)
            if key is None:
                print(f"[Chain verify] Unknown signer: {cur.signer_id}")
                return False
            header = canonical_json(cur.header_dict())
            if not verify_hmac_signature(key, header, cur.signature):
                print(f"[Chain verify] BAD signature at index {i} (signer {cur.signer_id})")
                return False
        return True

    def export_block_header(self, block_index: int) -> dict:
        block = self.chain[block_index]
        return block.to_json()

    def import_block_header(self, header: dict) -> None:
        block = Block(
            index=header["index"],
            prev_hash=header["prev_hash"],
            evidence_hash=header["evidence_hash"],
            metadata=header["metadata"],
            signer_id=header["signer_id"],
            timestamp=header["timestamp"],
            signature=header["signature"]
        )
        self.chain.append(block)


def demo_cross_border_sharing():
    agency_keys = {
        "AgencyA": b"agencyA_shared_secret_32bytes!",
        "AgencyB": b"agencyB_shared_secret_32bytes!",
    }

    chain_A = SimplePermissionedChain()
    chain_B = SimplePermissionedChain()

    raw_post = {
        "text": "Limited-time: click here to sign the petition to save the nation!",
        "author": "@some_account",
        "timestamp": "2025-09-21T19:12:00Z",
        "platform": "X",
    }

    analysis = {
        "ai_generated": True,
        "attribution": "GPT-family (confidence=0.81)",
        "harmful": True,
        "risk_score": 92
    }

    evidence_bundle = {
        "post": raw_post,
        "analysis": analysis,
        "collected_by": "AgencyA",
        "collected_at": "2025-09-21T19:20:00Z"
    }

    evidence_hash = create_evidence_hash(evidence_bundle)
    print("Agency A evidence hash:", evidence_hash)

    shared_metadata = {
        "platform": raw_post["platform"],
        "approx_time": raw_post["timestamp"],
        "attribution_summary": analysis["attribution"],
        "harmful": analysis["harmful"],
        "risk_score": analysis["risk_score"],
        "note": "Detected coordinated campaign signature; further checks recommended"
    }

    block_A = chain_A.add_signed_block(evidence_hash=evidence_hash, metadata=shared_metadata, signer_id="AgencyA", signer_key=agency_keys["AgencyA"])
    print("Agency A appended block index:", block_A.index)

    header_to_share = chain_A.export_block_header(block_A.index)
    chain_B.import_block_header(header_to_share)
    print("Agency B imported block header index:", header_to_share["index"])

    trusted_keys = {"AgencyA": agency_keys["AgencyA"], "AgencyB": agency_keys["AgencyB"]}
    okA = chain_B.verify_chain(trusted_keys=trusted_keys)
    print("Agency B chain verification result:", okA)

    provided_evidence = evidence_bundle
    recomputed_hash = create_evidence_hash(provided_evidence)
    print("Recomputed hash from provided evidence:", recomputed_hash)
    print("Matches stored hash?", recomputed_hash == evidence_hash)

    matches_chain = (recomputed_hash == chain_B.chain[-1].evidence_hash)
    print("Matches chain record at B?", matches_chain)

    print("\nShared block header (safe to share across borders):")
    print(json.dumps(header_to_share, indent=2))


demo_cross_border_sharing()
