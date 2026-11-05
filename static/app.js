const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');
const chipsEl = document.getElementById('chips');
const badge = document.getElementById('statusBadge');

const QUICK_QUESTIONS = [
  'What courses does your institute offer?',
  'Are your courses suitable for beginners?',
  'How much do courses cost?',
  'How do I enroll?',
  'Do I get a certificate?',
  'Do you provide job placement assistance?',
  'How can I get my doubts cleared?',
  'Can I attend a free demo class?',
];

function scrollBottom(){
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderRow(kind, text, meta){
  const row = document.createElement('div');
  row.className = 'row ' + kind;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  if (meta){
    const m = document.createElement('div');
    m.className = 'meta';
    m.textContent = meta;
    bubble.appendChild(m);
  }
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  scrollBottom();
  return row;
}

function setLoading(on){
  sendBtn.disabled = on;
  inputEl.disabled = on;
}

function buildChips(){
  QUICK_QUESTIONS.forEach(q=>{
    const btn = document.createElement('button');
    btn.className = 'chip';
    btn.textContent = q;
    btn.onclick = ()=>{ inputEl.value = q; sendMessage(); };
    chipsEl.appendChild(btn);
  });
}

async function healthCheck(){
  try{
    const r = await fetch('/api/health');
    const j = await r.json();
    if (j.groq_configured){
      badge.textContent = '● ' + j.model + ' live';
      badge.className = 'badge ok';
    } else {
      badge.textContent = '● FAQ fallback (no Groq key)';
      badge.className = 'badge warn';
    }
  }catch{
    badge.textContent = '● API offline';
    badge.className = 'badge warn';
  }
}

async function sendMessage(){
  const text = inputEl.value.trim();
  if (!text) return;
  renderRow('user', text);
  inputEl.value = '';
  setLoading(true);
  const typingRow = renderRow('bot', 'Thinking…');
  try{
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({query: text})
    });
    const data = await r.json();
    const meta = data.source ? 'Source: ' + data.source : '';
    typingRow.querySelector('.bubble').textContent = data.answer;
    if (meta){
      const m = document.createElement('div');
      m.className = 'meta';
      m.textContent = meta;
      typingRow.querySelector('.bubble').appendChild(m);
    }
  }catch(e){
    typingRow.querySelector('.bubble').textContent = 'Sorry, the server is unreachable. Is the backend running? (python main.py)';
  }finally{
    setLoading(false);
    inputEl.focus();
  }
}

buildChips();
healthCheck();
renderRow('bot',
  "Hi! I'm the LearnUp Academy FAQ assistant. Ask me about courses, fees, enrollment, certificates, or placement support.\n\nTip: Try one of the quick questions above."
);

inputEl.addEventListener('keydown', e=>{
  if (e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); sendMessage(); }
});
sendBtn.addEventListener('click', sendMessage);
