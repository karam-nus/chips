/* ============================================================
   Chips Guide — site chrome (vanilla JS, no dependencies)
   - reading progress trace along the top edge
   - sidebar chapter filter ("/" to focus, Esc to clear)
   - copy-to-clipboard on code blocks
   Every feature degrades to plain HTML when JS is off.
   ============================================================ */
(function () {
  'use strict';

  // ---------------------------------------------------------- reading progress
  var bar = document.getElementById('readProgress');
  if (bar) {
    var ticking = false;
    var update = function () {
      var h = document.documentElement;
      var max = h.scrollHeight - h.clientHeight;
      bar.style.width = (max > 0 ? Math.min(100, (h.scrollTop / max) * 100) : 0) + '%';
      ticking = false;
    };
    window.addEventListener('scroll', function () {
      if (!ticking) { ticking = true; window.requestAnimationFrame(update); }
    }, { passive: true });
    update();
  }

  // ---------------------------------------------------------- nav filter
  var input = document.getElementById('navFilter');
  var nav = document.getElementById('sidebarNav');
  if (input && nav) {
    var items = Array.prototype.slice.call(nav.children);
    var apply = function () {
      var q = input.value.trim().toLowerCase();
      var section = null, sectionHasMatch = false;
      var flush = function () { if (section) section.classList.toggle('filtered-out', q !== '' && !sectionHasMatch); };
      items.forEach(function (el) {
        if (el.classList.contains('nav-section')) {
          flush(); section = el; sectionHasMatch = false;
          return;
        }
        var hit = q === '' || el.textContent.toLowerCase().indexOf(q) !== -1;
        el.classList.toggle('filtered-out', !hit);
        if (hit) sectionHasMatch = true;
      });
      flush();
    };
    input.addEventListener('input', apply);
    input.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') { input.value = ''; apply(); input.blur(); }
      if (e.key === 'Enter') {
        var first = nav.querySelector('.nav-item:not(.filtered-out)');
        if (first) window.location.href = first.href;
      }
    });
    document.addEventListener('keydown', function (e) {
      var t = e.target && e.target.tagName;
      if (e.key === '/' && t !== 'INPUT' && t !== 'TEXTAREA' && t !== 'SELECT') {
        e.preventDefault();
        var layout = document.getElementById('layout');
        if (layout && layout.classList.contains('sidebar-collapsed')) {
          var tg = document.getElementById('sidebarToggle'); if (tg) tg.click();
        }
        input.focus();
      }
    });
  }

  // ---------------------------------------------------------- copy buttons
  var blocks = document.querySelectorAll('.content div.highlighter-rouge');
  Array.prototype.forEach.call(blocks, function (block) {
    var code = block.querySelector('td.rouge-code pre') || block.querySelector('pre');
    if (!code) return;
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'copy-btn';
    btn.textContent = 'Copy';
    btn.setAttribute('aria-label', 'Copy code to clipboard');
    btn.addEventListener('click', function () {
      var text = code.innerText.replace(/\n$/, '');
      var done = function () {
        btn.textContent = 'Copied'; btn.classList.add('done');
        setTimeout(function () { btn.textContent = 'Copy'; btn.classList.remove('done'); }, 1400);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () {});
      } else {
        var ta = document.createElement('textarea');
        ta.value = text; document.body.appendChild(ta); ta.select();
        try { document.execCommand('copy'); done(); } catch (err) { /* ignore */ }
        document.body.removeChild(ta);
      }
    });
    block.appendChild(btn);
  });
})();
