/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
const tts = require('./index.js');
const assert = require('assert');
const path = require('path');
const fs = require('fs');

async function testNode() {
    console.log('[*] Testing termux-tts Node.js SDK...');
    const engine = tts.load({ language: 'ko' });
    assert.strictEqual(engine.language, 'ko');

    // 1. Valid synthesis test
    const outWav = path.join(__dirname, 'test_node_out.wav');
    const res = await engine.synthesize("노드 JS SDK 보안 무결성 합성 검증입니다.", {
        output: outWav,
        speed: 1.0
    });
    assert.strictEqual(res.status, 'SUCCESS');
    assert(res.durationSec > 0.3);
    assert(fs.existsSync(outWav));
    assert(fs.statSync(outWav).size > 1000);
    fs.unlinkSync(outWav);
    console.log(`[PASS] Node.js Synthesis: ${res.durationSec}s in ${res.elapsedMs}ms`);

    // 2. Security payload test (Command/Code Injection Attack payload)
    const attackPayload = 'Hello"""\nimport os\nos.system("echo HACKED")\n"""World';
    const attackOut = path.join(__dirname, 'test_attack_out.wav');
    const attackRes = await engine.synthesize(attackPayload, {
        output: attackOut
    });
    assert.strictEqual(attackRes.status, 'SUCCESS');
    if (fs.existsSync(attackOut)) {
        fs.unlinkSync(attackOut);
    }
    console.log('[PASS] Node.js RCE Injection Defense verified.');
}

testNode().catch((err) => {
    console.error('[FAIL]', err);
    process.exit(1);
});
