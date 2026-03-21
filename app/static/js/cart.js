// ============================================
// cart.js — Shopping Cart Logic
// ============================================

const Cart = {

    // Get cart from localStorage
    get() {
        return JSON.parse(localStorage.getItem('KrishiSetu_cart') || '[]');
    },

    // Save cart to localStorage
    save(cart) {
        localStorage.setItem('KrishiSetu_cart', JSON.stringify(cart));
        this.updateBadge();
    },

    // Add item to cart
    add(id, name, price, image, category) {
        const cart = this.get();
        const existing = cart.find(item => item.id === id);
        if (existing) {
            existing.qty += 1;
        } else {
            cart.push({ id, name, price, image, category, qty: 1 });
        }
        this.save(cart);
        this.showToast(`${image} ${name} added to cart!`);
    },

    // Remove item
    remove(id) {
        const cart = this.get().filter(item => item.id !== id);
        this.save(cart);
        this.renderCart();
    },

    // Update quantity
    updateQty(id, delta) {
        const cart = this.get();
        const item = cart.find(i => i.id === id);
        if (item) {
            item.qty += delta;
            if (item.qty <= 0) {
                this.remove(id);
                return;
            }
        }
        this.save(cart);
        this.renderCart();
    },

    // Get total count
    count() {
        return this.get().reduce((sum, item) => sum + item.qty, 0);
    },

    // Get total price
    total() {
        return this.get().reduce(
            (sum, item) => sum + (item.price * item.qty), 0
        );
    },

    // Update cart badge in navbar
    updateBadge() {
        const badges = document.querySelectorAll('.cart-badge');
        const count  = this.count();
        badges.forEach(b => {
            b.textContent = count;
            b.style.display = count > 0 ? 'flex' : 'none';
        });
    },

    // Render cart sidebar
    renderCart() {
        const cart     = this.get();
        const container = document.getElementById('cartItems');
        const totalEl   = document.getElementById('cartTotal');
        const countEl   = document.getElementById('cartCount');

        if (!container) return;

        this.updateBadge();

        if (countEl) countEl.textContent = this.count();

        if (cart.length === 0) {
            container.innerHTML = `
                <div class="cart-empty">
                    <div style="font-size:3rem;margin-bottom:12px;">🛒</div>
                    <p>Your cart is empty</p>
                    <a href="/shop" style="color:#27ae60;font-size:0.9rem;">
                        Browse products →
                    </a>
                </div>`;
            if (totalEl) totalEl.textContent = '₹0';
            return;
        }

        container.innerHTML = cart.map(item => `
            <div class="cart-item">
                <span class="cart-item-img">${item.image}</span>
                <div class="cart-item-info">
                    <div class="cart-item-name">${item.name}</div>
                    <div class="cart-item-price">₹${item.price}</div>
                </div>
                <div class="cart-item-controls">
                    <button onclick="Cart.updateQty('${item.id}', -1)">−</button>
                    <span>${item.qty}</span>
                    <button onclick="Cart.updateQty('${item.id}', 1)">+</button>
                </div>
                <button class="cart-item-remove"
                        onclick="Cart.remove('${item.id}')">✕</button>
            </div>
        `).join('');

        if (totalEl) totalEl.textContent = `₹${this.total()}`;
    },

    // Toggle cart sidebar
    toggle() {
        const sidebar = document.getElementById('cartSidebar');
        const overlay = document.getElementById('cartOverlay');
        if (sidebar) {
            sidebar.classList.toggle('open');
            overlay.classList.toggle('active');
            this.renderCart();
        }
    },

    // Show toast notification
    showToast(message) {
        const existing = document.getElementById('cartToast');
        if (existing) existing.remove();

        const toast = document.createElement('div');
        toast.id = 'cartToast';
        toast.className = 'cart-toast';
        toast.textContent = message;
        document.body.appendChild(toast);

        setTimeout(() => toast.classList.add('show'), 10);
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 2500);
    },

    // Checkout
    checkout() {
        if (this.count() === 0) {
            alert('Your cart is empty!');
            return;
        }
        alert(`Order placed successfully!\nTotal: ₹${this.total()}\nThank you for shopping with KrishiSetu!`);
        localStorage.removeItem('KrishiSetu_cart');
        this.updateBadge();
        this.renderCart();
        this.toggle();
    }
};

// Init badge on page load
document.addEventListener('DOMContentLoaded', () => Cart.updateBadge());