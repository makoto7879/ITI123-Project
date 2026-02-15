import gradio as gr
import torch
import torch.nn as nn
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK data with proper error handling
print("Downloading NLTK data...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("✓ NLTK data downloaded successfully")
except Exception as e:
    print(f"Warning: NLTK download issue: {e}")

# Verify downloads
try:
    word_tokenize("test")
    stopwords.words('english')
    print("✓ NLTK is working correctly")
except Exception as e:
    print(f"✗ NLTK verification failed: {e}")


# Model Definition
class ImprovedSPAMGuru(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_size=256, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            embed_dim, 
            hidden_size, 
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Classification layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, _ = self.gru(embedded)
        attention_weights = torch.softmax(self.attention(gru_out), dim=1)
        context = torch.sum(attention_weights * gru_out, dim=1)
        x = self.fc1(context)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x).squeeze()


# Text preprocessing functions (matching your latest notebook exactly)
def clean_text(text):
    """
    Advanced text cleaning with URL extraction, money detection, phishing detection, and prize scam detection
    """
    # 1. Decode HTML entities & remove tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # 2. Extract URL domain features (BEFORE phishing detection)
    def extract_url_features(text):
        """Extract domain components"""
        urls = re.findall(r'(http[s]?://[^\s]+|www\.[^\s]+)', text, flags=re.IGNORECASE)
        
        for url in urls:
            domain_match = re.search(r'(?:http[s]?://)?([^/\s]+)', url)
            
            if domain_match:
                domain = domain_match.group(1)
                domain = re.sub(r'^www\.', '', domain, flags=re.IGNORECASE)
                domain_parts = domain.replace('.', ' ').replace('-', ' ')
                replacement = f' url {domain_parts} '
                text = text.replace(url, replacement)
        
        return text
    
    text = extract_url_features(text)
    
    # 3. Email and phone replacement
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)
    text = re.sub(r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b', ' PHONE ', text)
    
    # 4. Detect large money amounts BEFORE lowercasing
    amounts = re.findall(r'\$[\d,]+', text)
    for amount in amounts:
        num_str = amount.replace('$', '').replace(',', '')
        if num_str.isdigit():
            num = int(num_str)
            if num >= 10000:
                text = text.replace(amount, ' XXLARGEMONEY ')
            elif num >= 100:
                text = text.replace(amount, ' XXMONEY ')
    
    # 5. Lowercase
    text = text.lower()
    
    # 6. Replace money placeholders with lowercase tokens
    text = text.replace('xxlargemoney', 'largemoney')
    text = text.replace('xxmoney', 'money')
    
    # 7. Phishing detection (after lowercasing, before prize detection)
    phishing_words = ['verify', 'suspended', 'confirm', 'urgent', 'action required',
                      'update payment', 'security alert', 'unusual activity']
    has_url = bool(re.search(r'url', text))
    has_phishing_word = any(word in text for word in phishing_words)
    
    if has_url and has_phishing_word:
        text = 'phishing ' + text
    
    # 8. Prize scam detection (after money and lowercasing)
    prize_words = ['won', 'win', 'winner', 'prize', 'congratulations', 'claim']
    has_large_money = 'largemoney' in text
    has_prize_word = any(word in text for word in prize_words)
    
    if has_large_money and has_prize_word:
        text = 'prizescam ' + text
    
    # 9. Normalize repeated punctuation
    text = re.sub(r'([!?.]){3,}', r'\1\1\1', text)
    
    # 10. Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize(text):
    """
    Tokenize with light stopword removal
    """
    stop_words = set(stopwords.words('english')) - {'not', 'no', 'never', 'none'}
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in list(stop_words)[:50]]
    return tokens


def encode(text, stoi, max_len=200):
    """
    Encode text to sequence of indices
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    ids = [stoi.get(token, stoi.get("<UNK>", 1)) for token in tokens]
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids = ids + [stoi.get("<PAD>", 0)] * (max_len - len(ids))
    return ids


# Load model and config
print("="*60)
print("Loading ImprovedSPAMGuru Model...")
print("="*60)

try:
    config = torch.load('spam_config.pth', map_location='cpu', weights_only=False)
    print(f"✓ Configuration loaded")
    print(f"  Vocab size: {config['vocab_size']:,}")
    print(f"  Embed dim: {config['embed_dim']}")
    print(f"  Hidden size: {config['hidden_size']}")
    print(f"  Max length: {config['max_len']}")
    
    model = ImprovedSPAMGuru(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        hidden_size=config['hidden_size'],
        dropout=config['dropout']
    )
    
    model.load_state_dict(torch.load('spam_model.pth', map_location='cpu', weights_only=True))
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"✓ Model loaded successfully!")
    print("="*60)
    
except Exception as e:
    print(f"✗ ERROR loading model: {str(e)}")
    import traceback
    traceback.print_exc()
    raise


def predict_spam(text):
    """Predict if text is spam or ham with detailed analysis"""
    try:
        if not text or not text.strip():
            return {
                "⚠️ Error": 1.0,
                "Enter text to analyze": 0.0
            }, "Please enter an email to analyze."
        
        # Preprocess
        cleaned = clean_text(text)
        tokens = tokenize(cleaned)
        
        # Encode
        indices = encode(text, config['stoi'], config['max_len'])
        
        # Convert to tensor
        x = torch.tensor([indices])
        
        # Predict
        with torch.no_grad():
            probability = model(x).item()
        
        # Create result dictionary
        if probability >= 0.5:
            result = {
                "🚨 SPAM": float(probability),
                "✅ HAM": float(1 - probability)
            }
            verdict = "🚨 SPAM"
        else:
            result = {
                "✅ HAM": float(1 - probability),
                "🚨 SPAM": float(probability)
            }
            verdict = "✅ HAM"
        
        # Create analysis
        analysis = f"""### 📊 Analysis Results

**Verdict:** {verdict}  
**Confidence:** {max(probability, 1-probability)*100:.1f}%

**Preprocessing:**
- Cleaned text preview: {cleaned[:100]}{'...' if len(cleaned) > 100 else ''}
- Tokens extracted: {len(tokens)} words
- Sample tokens: {', '.join(tokens[:12])}{'...' if len(tokens) > 12 else ''}

**Detection Signals:**"""
        
        # Check for spam signals
        signals = []
        if 'prizescam' in cleaned.lower():
            signals.append("🎰 **PRIZE SCAM DETECTED** (won + large money)")
        if 'phishing' in cleaned.lower():
            signals.append("🎣 **PHISHING PATTERN DETECTED** (urgent + URL)")
        if 'largemoney' in cleaned.lower():
            signals.append("💰 Large money amount detected")
        elif 'money' in cleaned.lower() and 'largemoney' not in cleaned.lower():
            signals.append("💵 Money amount detected")
        if 'url' in cleaned:
            url_parts = [t for t in tokens if t not in ['url', 'com', 'net', 'org', 'xyz', 'click']]
            if url_parts:
                signals.append(f"🌐 URL detected: {' '.join(url_parts[:3])}")
            else:
                signals.append("🌐 URL detected")
        if any(word in tokens for word in ['win', 'won', 'prize', 'claim', 'congratulations', 'winner']):
            signals.append("🎁 Prize/winning keywords detected")
        if any(word in tokens for word in ['free', 'urgent', 'limited', 'act', 'now']):
            signals.append("⚡ Urgency keywords detected")
        if any(word in tokens for word in ['verify', 'suspended', 'confirm', 'action', 'required']):
            signals.append("⚠️ Suspicious action keywords detected")
        
        if signals:
            for signal in signals:
                analysis += f"\n- {signal}"
        else:
            analysis += "\n- ✓ No obvious spam signals detected"
        
        return result, analysis
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "⚠️ Error": 1.0
        }, f"Error: {str(e)}"


# Example emails
spam_1 = "You've just won $2,500,000! Claim now: http://luckyclaim.net"
spam_2 = "CONGRATULATIONS!!! FREE iPhone 15! Click: http://winner-prize.com LIMITED TIME!!!"
spam_3 = "URGENT: Account suspended! Verify now: http://secure-bank-login.xyz"
spam_4 = "Security Alert: Unusual activity detected. Confirm your identity: http://verify-account.com"
ham_1 = "Hi team, meeting tomorrow at 2 PM in Conference Room B. Please review the documents. Best, John"
ham_2 = "Thanks for the update on the $2,000,000 Smith contract. I'll review and get back to you. Sarah"
ham_3 = "Weekly update: Project Alpha 80% complete. New hire starts Monday. See you at standup. Mike"


# Create Gradio interface
demo = gr.Blocks(title="Advanced Spam Detector - ImprovedSPAMGuru")

with demo:
    gr.Markdown(
        """
        # 📧 Advanced Email Spam Detector
        ### ImprovedSPAMGuru v2.0 - Now with Prize Scam & Phishing Detection! 🎣🎰
        
        **🎯 97.93% Base Accuracy** | **🔍 Advanced Pattern Recognition** | **🧠 Deep Learning**
        
        ✨ **Features:**
        - 🎰 Prize scam detection ("You won $X million!")
        - 🎣 Phishing detection (account verification scams)
        - 💰 Money amount analysis
        - 🌐 URL domain extraction
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            email_input = gr.Textbox(
                label="📝 Enter Email Text",
                placeholder="Paste email content here...\n\nDetects:\n- Prize scams ($1M+ offers)\n- Phishing attacks (verify account)\n- Suspicious URLs\n- Urgency tactics",
                lines=12,
                max_lines=20
            )
            
            with gr.Row():
                submit_btn = gr.Button("🔍 Analyze Email", variant="primary", scale=2)
                clear_btn = gr.ClearButton([email_input], value="Clear", scale=1)
        
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Try Examples")
            gr.Markdown("**🔴 Spam:**")
            spam_btn_1 = gr.Button("💰 Prize Scam ($2.5M)", size="sm")
            spam_btn_2 = gr.Button("📱 Free iPhone", size="sm")
            spam_btn_3 = gr.Button("🎣 Phishing (Urgent)", size="sm")
            spam_btn_4 = gr.Button("⚠️ Security Alert", size="sm")
            
            gr.Markdown("**🟢 Legitimate:**")
            ham_btn_1 = gr.Button("📅 Meeting", size="sm")
            ham_btn_2 = gr.Button("💼 Business Contract", size="sm")
            ham_btn_3 = gr.Button("📊 Team Update", size="sm")
            
            gr.Markdown(
                """
                ### ℹ️ Model Info
                - **Base Accuracy:** 97.93%
                - **Spam Detection:** 98.95%
                - **Architecture:** BiGRU + Attention
                - **Vocabulary:** 15,000 words
                - **Special Features:**
                  - 🎰 Prize scam detection
                  - 🎣 Phishing detection
                  - 💰 Money analysis
                """
            )
    
    gr.Markdown("---")
    
    with gr.Row():
        output_label = gr.Label(label="🎯 Classification", num_top_classes=2)
        output_analysis = gr.Markdown(value="*Submit email to see detailed analysis*")
    
    with gr.Accordion("💡 Detection Capabilities", open=False):
        gr.Markdown(
            """
            ### 🎰 Prize Scam Detection
            
            Detects combinations of:
            - 💰 Large money amounts ($10,000+)
            - 🎁 Prize/winning language ("won", "prize", "claim")
            - 🔗 Suspicious URLs
            
            **Examples:**
            - ✅ "You won $2,500,000! Claim at winner.com"
            - ✅ "CONGRATULATIONS! $1M prize! Click here!"
            
            ### 🎣 Phishing Detection
            
            Detects combinations of:
            - ⚠️ Urgent action words ("verify", "suspended", "urgent")
            - 🔗 URLs (often fake domains)
            - 🎭 Impersonation attempts
            
            **Examples:**
            - ✅ "URGENT: Account suspended! Verify now"
            - ✅ "Security alert: Confirm your identity"
            
            ### 🛡️ How It Works
            
            1. **Advanced Preprocessing:** Extracts domain names, money amounts, patterns
            2. **Special Tokens:** Creates explicit signals for spam types
            3. **Deep Learning:** Bidirectional GRU with attention mechanism
            4. **Context Analysis:** Understands word combinations and relationships
            
            ### 📊 Limitations
            
            - Optimized for English language emails
            - May flag legitimate marketing emails
            - New spam patterns may require retraining
            - Short messages (<10 words) may be harder to classify
            """
        )
    
    # Event handlers
    submit_btn.click(predict_spam, inputs=email_input, outputs=[output_label, output_analysis])
    spam_btn_1.click(lambda: spam_1, outputs=email_input)
    spam_btn_2.click(lambda: spam_2, outputs=email_input)
    spam_btn_3.click(lambda: spam_3, outputs=email_input)
    spam_btn_4.click(lambda: spam_4, outputs=email_input)
    ham_btn_1.click(lambda: ham_1, outputs=email_input)
    ham_btn_2.click(lambda: ham_2, outputs=email_input)
    ham_btn_3.click(lambda: ham_3, outputs=email_input)
    
    gr.Markdown(
        """
        ---
        <div style='text-align: center; color: gray;'>
        <b>ImprovedSPAMGuru v2.0</b> - Prize Scam & Phishing Detection 🎣🎰 | 
        Dataset: Spam Assassin (5,796 emails) | Base Accuracy: 97.93%
        </div>
        """
    )

demo.launch(ssr_mode=False)