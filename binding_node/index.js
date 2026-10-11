/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/**
 * termux-tts Node.js SDK
 * Secure On-Device Speech Synthesis Binding (CWE-94 Remediation).
 */

const { spawn } = require('child_process');
const path = require('path');

class TTSEngine {
    constructor(options = {}) {
        this.language = options.language || 'ko';
        this.preset = options.preset || 'balanced';
        this.sampleRate = options.sampleRate || 22050;
        this.engine = options.engine || 'auto';
        this.device = options.device || (options.gpu ? 'gpu' : 'auto');
        this.tier = options.tier || null;
        this.model = options.model || null;
    }

    async synthesize(text, options = {}) {
        if (!text || typeof text !== 'string' || !text.trim()) {
            throw new Error('TTSInferenceError: Input text cannot be empty.');
        }

        const output = options.output || path.join(process.cwd(), 'output.wav');
        const speed = typeof options.speed === 'number' ? options.speed : 1.0;
        const preset = options.preset || this.preset;
        const engineType = options.engine || this.engine;
        const deviceType = options.device || (options.gpu ? 'gpu' : this.device);
        const tierType = options.tier || this.tier;
        const modelPath = options.model || this.model;

        const payload = JSON.stringify({
            text: text,
            output: output,
            language: this.language,
            sample_rate: this.sampleRate,
            speed: speed,
            preset: preset,
            engine: engineType,
            device: deviceType,
            tier: tierType,
            model: modelPath
        });

        return new Promise((resolve, reject) => {
            const pythonDriver = `
import sys, json
import termux_tts as tts

try:
    data = json.loads(sys.stdin.read())
    engine = tts.load(
        model=data.get('model'),
        language=data.get('language', 'ko'),
        preset=data.get('preset', 'balanced'),
        sample_rate=data.get('sample_rate', 22050),
        engine=data.get('engine', 'auto'),
        device=data.get('device', 'auto'),
        tier=data.get('tier', None)
    )
    res = engine.synthesize(data['text'], output=data['output'], speed=float(data.get('speed', 1.0)))
    result = {
        'status': 'SUCCESS',
        'text': res.text,
        'outputPath': data['output'],
        'durationSec': res.duration_sec,
        'elapsedMs': res.elapsed_ms,
        'rtf': res.rtf,
        'sampleRate': res.sample_rate,
        'backend': res.backend,
        'modelName': getattr(res, 'model_name', 'default')
    }
    sys.stdout.write(json.dumps(result))
except Exception as e:
    sys.stderr.write(f"TTSInferenceError: {str(e)}")
    sys.exit(1)
`;
            const pyBin = process.env.PYTHON_BIN || (process.platform === 'win32' ? 'py' : 'python3');
            const proc = spawn(pyBin, ['-c', pythonDriver], {
                env: { ...process.env, PYTHONPATH: path.join(__dirname, '..') }
            });

            let stdout = '';
            let stderr = '';

            proc.stdout.on('data', (d) => { stdout += d.toString(); });
            proc.stderr.on('data', (d) => { stderr += d.toString(); });

            proc.on('close', (code) => {
                if (code !== 0) {
                    return reject(new Error(`TTSInferenceError (${code}): ${stderr.trim() || stdout.trim()}`));
                }
                try {
                    const parsed = JSON.parse(stdout);
                    if (parsed.status === 'SUCCESS') {
                        resolve(parsed);
                    } else {
                        reject(new Error(`TTSInferenceError: Unexpected execution status in ${stdout}`));
                    }
                } catch (err) {
                    reject(new Error(`TTSParseError: Failed to parse Python worker JSON response: ${stdout}`));
                }
            });

            proc.on('error', (err) => {
                reject(new Error(`TTSProcessError: Failed to spawn python3 worker: ${err.message}`));
            });

            // Write JSON payload securely to stdin and close stream
            proc.stdin.write(payload);
            proc.stdin.end();
        });
    }

    close() {
        // Resource clean up
    }
}

module.exports = {
    TTSEngine,
    load: (opts) => new TTSEngine(opts),
    version: "0.1.0"
};
