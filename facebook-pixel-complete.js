// ==================================================
// FACEBOOK PIXEL + CAPI GATEWAY - RUBYWINGS.VN
// ==================================================

// Load Facebook Pixel Base Code
(function() {
    if (window.fbqLoaded) return;
    window.fbqLoaded = true;
    
    !function(f,b,e,v,n,t,s)
    {if(f.fbq)return;n=f.fbq=function(){n.callMethod?
    n.callMethod.apply(n,arguments):n.queue.push(arguments)};
    if(!f._fbq)f._fbq=n;n.push=n;n.loaded=!0;n.version='2.0';
    n.queue=[];t=b.createElement(e);t.async=!0;
    t.src=v;s=b.getElementsByTagName(e)[0];
    s.parentNode.insertBefore(t,s)}(window, document,'script',
    'https://connect.facebook.net/en_US/fbevents.js');
    
    // Initialize Pixel với CAPI Gateway
    fbq('init', '862531473384426', {
        external_id: getVisitorID(),
        tn: 'CAPI Gateway'
    });
    fbq('track', 'PageView');
    
    // CAPI Gateway endpoint CHO NETLIFY
  fbq('set', 'bridge', 'https://www.rubywings.vn/capi', '862531473384426');
})();

// Hàm lấy Visitor ID cho CAPI
function getVisitorID() {
    let visitorId = localStorage.getItem('rw_visitor_id');
    if (!visitorId) {
        visitorId = 'rw_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('rw_visitor_id', visitorId);
    }
    return visitorId;
}

// CAPI Enhanced Tracking Functions
function trackConsultationClick() {
    const visitorData = {
        external_id: getVisitorID(),
        client_ip_address: '',
        client_user_agent: navigator.userAgent,
        event_source_url: window.location.href,
        action_source: 'website'
    };
    
    fbq('track', 'Lead', {
        content_name: 'Đăng ký tư vấn chung - ' + document.title,
        content_category: 'General Consultation',
        page_url: window.location.href,
        button_type: 'General CTA'
    }, visitorData);
    
    // Google Analytics tracking
    if (typeof gtag !== 'undefined') {
        gtag('event', 'consultation_click', {
            'event_category': 'engagement',
            'event_label': 'Google Forms Consultation Button'
        });
    }
    
    console.log('Facebook Pixel: Tracked consultation click');
}

function trackTourRegistration(tourName, tourCategory) {
    const visitorData = {
        external_id: getVisitorID(),
        client_ip_address: '',
        client_user_agent: navigator.userAgent,
        event_source_url: window.location.href,
        action_source: 'website'
    };
    
    fbq('track', 'Lead', {
        content_name: 'Đăng ký: ' + tourName,
        content_category: tourCategory,
        content_ids: [tourName],
        page_url: window.location.href,
        button_type: 'Tour Specific CTA'
    }, visitorData);
    
    console.log('Facebook Pixel: Tracked tour registration - ' + tourName);
}

// Track Page Views for Specific Tours
function trackTourView(tourName, tourCategory) {
    const visitorData = {
        external_id: getVisitorID(),
        client_ip_address: '',
        client_user_agent: navigator.userAgent,
        event_source_url: window.location.href,
        action_source: 'website'
    };
    
    fbq('track', 'ViewContent', {
        content_name: tourName,
        content_category: tourCategory,
        content_ids: [tourName],
        content_type: 'product'
    }, visitorData);
}

// Auto-detect and track tour pages
document.addEventListener('DOMContentLoaded', function() {
    const url = window.location.href;
    
    if (url.includes('du-lich-trai-nghiem-cam-xuc')) {
        trackTourView('Du lịch Trải nghiệm Cảm xúc', 'Experience Tourism');
    }
    else if (url.includes('du-lich-chua-lanh')) {
        trackTourView('Du lịch Chữa lành', 'Healing Tourism');
    }
    else if (url.includes('retreat-thien')) {
        trackTourView('Retreat Thiền và Khí công', 'Wellness Retreat');
    }
    else if (url.includes('team-building')) {
        trackTourView('Team Building Trải nghiệm', 'Corporate Events');
    }
});

// Backup tracking
function ensurePixelTracking() {
    if (typeof fbq === 'undefined') {
        setTimeout(ensurePixelTracking, 500);
    }
}
ensurePixelTracking();

// Noscript fallback
document.write('<noscript><img height="1" width="1" style="display:none" src="https://www.facebook.com/tr?id=862531473384426&ev=PageView&noscript=1"/></noscript>');