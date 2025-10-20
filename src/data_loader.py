import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
import re
from tqdm import tqdm
import json
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
import pytz

@dataclass
class Email:
    id: str
    message_id: str
    date: Optional[str]
    from_address: str
    to_addresses: List[str]
    cc_addresses: List[str] = None
    bcc_addresses: List[str] = None
    subject: str = ""
    body: str = ""
    content_type: Optional[str] = ""
    encoding: Optional[str] = ""
    x_from: Optional[str] = ""
    x_to: Optional[str] = "" 
    x_cc: Optional[str] = "" 
    x_bcc: Optional[str] = "" 
    x_folder: Optional[str] = "" 
    x_origin: Optional[str] = "" 
    x_filename: Optional[str] = "" 

    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def get_full_text(self) -> str:
        parts = []
        if self.subject:
            parts.append(f"Subject: {self.subject}")
        if self.from_address:
            parts.append(f"From: {self.from_address}")
        if self.to_addresses:
            parts.append(f"To: {', '.join(self.to_addresses)}")
        if self.body:
            parts.append(f"\nContent:\n{self.body}")
        return "\n".join(parts)
    
    def get_metadata(self) -> Dict:
        return {
            "id": self.id,
            "message_id": self.message_id,
            "date": self.date,
            "from": self.from_address,
            "to": ", ".join(self.to_addresses) if self.to_addresses else "",
            "subject": self.subject,
            "thread_id": self.thread_id
        }


class EnronDataLoader:
    
    def __init__(self, csv_path: str, sample_size: Optional[int] = None):

        self.csv_path = Path(csv_path)
        self.sample_size = sample_size
        self.emails: List[Email] = []
        
        self.email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
        self.thread_pattern = re.compile(r'(RE:|FW:|Fwd:|Re:)', re.IGNORECASE)
        
    def load_emails(self) -> List[Email]:
        
        try:
            df = pd.read_csv(
                self.csv_path,
                nrows=self.sample_size,
                encoding='utf-8',
                on_bad_lines='skip'
            )

            # df.to_csv('datos_procesados.csv', index=False, encoding='utf-8')
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Procesando emails"):
                email = self._process_email_row(row, idx)
                if email and email.body: 
                    self.emails.append(email)
            print(f"{len(self.emails)} emails válidos")
            return self.emails
            
        except Exception as e:
            print(f"Error cargando CSV: {e}")
            raise
    
    def _process_email_row(self, row: pd.Series, idx: int) -> Optional[Email]:

        try:
            body_raw = str(row.get('message', '')).strip()
            if not body_raw:
                return None
            if '\n\n' in body_raw:
                header_block, body = body_raw.split('\n\n', 1)
            else:
                header_block, body = body_raw, ''
            
            headers = {}
            last_key = None
            
            for line in header_block.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1) 
                    key = key.replace("-", "_").lower().strip()
                    if key not in headers: 
                        headers[key] = value.strip()
                        last_key = key
                elif last_key and (line.startswith(' ') or line.startswith('\t')):
                    headers[last_key] += ' ' + line.strip()

            headers["body"] = body.strip()


            date_iso = self._process_date(headers.get("date", ""))

            def parse_addresses(raw: str):
                return [a.strip() for a in re.split(r",|;", raw) if a.strip()]
            from_addr = headers.get("from", "")
            to_addrs = parse_addresses(headers.get("to", ""))
            x_cc_addrs = parse_addresses(headers.get("x_cc", ""))
            x_bcc_addrs = parse_addresses(headers.get("x_bcc", ""))

            email_id = hashlib.md5((from_addr + headers.get("subject","") + date_iso + str(idx)).encode()).hexdigest()[:12]

            email = Email(
                id=email_id,
                message_id=headers.get("message_id"),
                date=date_iso,
                from_address=from_addr,
                to_addresses=to_addrs,
                subject=headers.get("subject"),
                body=headers.get("body",""),
                content_type=headers.get("content_type"),
                encoding=headers.get("content_transfer_encoding"),
                x_from=headers.get("x_from"),
                x_to=headers.get("x_to"),
                x_cc=x_cc_addrs,
                x_bcc=x_bcc_addrs,
                x_folder=headers.get("x_folder"),
                x_origin=headers.get("x_origin"),
                x_filename=headers.get("x_filename"),
            )
            return email
        except Exception as e:
            print(f"Error procesando fila {idx}: {e}")
            return None

    def _process_date(self, date_str: str) :
        date_iso = None
        if date_str:
            try:
                parsed_date = dateparser.parse(date_str, fuzzy=True)
                if parsed_date.tzinfo:
                    parsed_date = parsed_date.astimezone(pytz.UTC)
                else:
                    parsed_date = pytz.UTC.localize(parsed_date)

                date_iso = parsed_date.isoformat()
                return date_iso
            except Exception as e:
                print(f"No se pudo parsear fecha '{date_str}': {e}")
                date_iso = date_str
                return date_iso
    
    def save_processed_emails(self, output_path: str):
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        emails_data = [email.to_dict() for email in self.emails]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(emails_data, f, indent=2, ensure_ascii=False)
        
        print(f"Emails guardados en {output_file}")
    
    def get_sample_for_testing(self, n: int = 100) -> List[Email]:
        if not self.emails:
            self.load_emails()
        
        if len(self.emails) <= n:
            return self.emails
        
        from_groups = {}
        for email in self.emails:
            sender = email.from_address
            if sender not in from_groups:
                from_groups[sender] = []
            from_groups[sender].append(email)
        
        sample = []
        senders = list(from_groups.keys())
        np.random.shuffle(senders)
        
        for sender in senders:
            if len(sample) >= n:
                break
            sample.extend(from_groups[sender][:max(1, n // len(senders))])
        
        return sample[:n]


def quick_test(csv_path: str):
    loader = EnronDataLoader(csv_path, sample_size=1000)
    emails = loader.load_emails()
    
    if emails:
        print(f"  - Total emails: {len(emails)}")
        print(f"  - Emails con subject: {sum(1 for e in emails if e.subject)}")
        print(f"  - Emails con body: {sum(1 for e in emails if e.body)}")
        print(f"  - Remitentes únicos: {len(set(e.from_address for e in emails))}")
        
        loader.save_processed_emails("../data/processed/enron_sample_1000.json")
    
    return emails


if __name__ == "__main__":
    csv_path = "../data/raw/emails.csv"
    emails = quick_test(csv_path)