function sendMessage() {
  const input = document.getElementById('user-input');
  const message = input.value.trim();
  if (message === '') return;

  const chatBody = document.getElementById('chat-body');

  // User message
  const userMsg = document.createElement('div');
  userMsg.className = 'message';
  userMsg.innerHTML = `<div class="text">${message}</div>`;
  chatBody.appendChild(userMsg);

  // Scroll to bottom
  chatBody.scrollTop = chatBody.scrollHeight;

  input.value = '';

  // Fake bot response
  setTimeout(() => {
    const botMsg = document.createElement('div');
    botMsg.className = 'message bot';
    botMsg.innerHTML = `<span class="bot-icon">🤖</span><div class="text">You said: "${message}"</div>`;
    chatBody.appendChild(botMsg);
    chatBody.scrollTop = chatBody.scrollHeight;
  }, 600);
}
