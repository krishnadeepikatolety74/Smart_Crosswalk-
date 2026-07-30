import os
import glob
import re

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, 'templates')
html_files = glob.glob(os.path.join(template_dir, '*.html'))

# Common patterns mapping
patterns = [
    # 1. Update old background gradients to new clean look
    (r'bg-gradient-to-br from-\[#fdfcff\] to-\[#f3e8ff\]', 'bg-theme-white'),
    
    # 2. Convert standard cards to Glassmorphism cards
    (r'bg-white/80 backdrop-blur-md p-6 rounded-2xl shadow-lg border border-\[#d6bcfa\]/50 transform transition hover:-translate-y-1', 
     'bg-white/60 backdrop-blur-2xl p-6 rounded-[24px] shadow-glass border border-white/60 hover:-translate-y-1 hover:shadow-glass-hover transition-all duration-300'),
    (r'bg-white/80 backdrop-blur-md p-6 rounded-2xl shadow-lg border border-\[#d6bcfa\]/50', 
     'bg-white/60 backdrop-blur-2xl p-6 rounded-[24px] shadow-glass border border-white/60 hover:-translate-y-1 hover:shadow-glass-hover transition-all duration-300'),
    (r'bg-white/80 p-6 rounded-2xl shadow-lg border border-\[#d6bcfa\]/50', 
     'bg-white/60 backdrop-blur-2xl p-6 rounded-[24px] shadow-glass border border-white/60 hover:-translate-y-1 hover:shadow-glass-hover transition-all duration-300'),
     
    # 3. Inner cards (like metrics inside live camera)
    (r'bg-white/60 p-4 rounded-lg border border-purple-100',
     'bg-white/50 backdrop-blur-lg p-5 rounded-2xl shadow-sm border border-white/80 hover:shadow-md transition-all duration-300'),
    (r'bg-white/60 p-4 rounded-lg border border-red-100',
     'bg-theme-coral/10 backdrop-blur-lg p-5 rounded-2xl shadow-sm border border-theme-coral/30 hover:shadow-md transition-all duration-300'),

    # 4. Buttons
    (r'bg-gradient-to-r from-\[#9f7aea\] to-\[#b794f4\] hover:from-\[#805ad5\] hover:to-\[#9f7aea\]',
     'bg-gradient-to-r from-theme-indigo to-theme-blue hover:from-theme-blue hover:to-theme-indigo shadow-md hover:shadow-lg transition-all'),
    (r'bg-\[#805ad5\] hover:bg-\[#6b46c1\]',
     'bg-theme-indigo hover:bg-theme-blue shadow-md hover:shadow-lg transition-all'),
     
    # 5. Badges / "LIVE" tags
    (r'bg-green-100 text-green-700 rounded-full text-sm font-semibold animate-pulse',
     'bg-theme-mint/20 text-green-800 rounded-full text-sm font-bold animate-pulse shadow-glow-live border border-theme-mint/50'),

    # 6. Sidebar active state
    (r'bg-\[#e9d5ff\]/50 text-\[#322659\]',
     'bg-theme-blue/20 text-theme-indigo shadow-[inset_0_2px_4px_rgba(0,0,0,0.05)] border border-theme-blue/30 rounded-xl'),

    # 7. Sidebar base
    (r'shadow-2xl bg-white/70 backdrop-blur-xl border-r border-\[#d6bcfa\]/50',
     'shadow-[4px_0_24px_rgba(110,139,255,0.1)] bg-white/40 backdrop-blur-3xl border-r border-white/60'),
     
    # 8. Text colors
    (r'text-purple-800', 'text-gray-800'),
    (r'text-purple-900', 'text-gray-900'),
    (r'text-\[#322659\]', 'text-gray-900'),
    (r'text-\[#44337a\]', 'text-gray-600 hover:text-theme-indigo transition-colors'),
    (r'text-\[#553c9a\]', 'text-theme-indigo'),
    (r'text-\[#805ad5\]', 'text-theme-lavender'),
    (r'border-\[#d6bcfa\]', 'border-theme-lavender/40'),
    (r'border-purple-200', 'border-theme-lavender/30'),
]

for filepath in html_files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    modified_content = content
    for pattern, replacement in patterns:
        modified_content = re.sub(pattern, replacement, modified_content)
        
    if modified_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        print(f"Updated {os.path.basename(filepath)}")
